import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from utils.load_graphdata import load_origindata_train, load_origindata_test, load_reframingsdata
from tqdm import tqdm
import warnings
from models.fourierattention import FourierAttention
from models.bert import RobertaClassifier
import logging
from datetime import datetime
from result_output.log_result import *
import json
#with open("config.json", "r") as f:
#    config = json.load(f)

warnings.filterwarnings("ignore")
device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str) #
parser.add_argument('--model_name', default='ERFD', type=str)
parser.add_argument('--iters', default=1, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--max_len', default = 512, type=int)
parser.add_argument('--hidden_dim', default = 768, type=int)
parser.add_argument('--freq_dim', default = 4, type=int)  #2,4,8,16,32,64,128,256,768
parser.add_argument('--attn_heads', default = 1, type = int) #1, 2, 4, 8, 16, 32
parser.add_argument('--loss_weight', default = 0.05, type = float) #0.1, 0.3, 0.5, 0.7, 1
parser.add_argument('--dropout_num', default = 0.4, type = float)
args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def test_ouput(test_loader,model):
    y_pred = []
    y_test = []
    for Batch_data in tqdm(test_loader):
        with torch.no_grad():
            input_ids = Batch_data["input_ids"].to(device)
            attention_mask = Batch_data["attention_mask"].to(device)
            targets = Batch_data["label"].to(device)
            val_out, _ = model(input_ids=input_ids, attention_masks=attention_mask)
            _, val_pred = val_out.max(dim=1)
            y_pred.append(val_pred)
            y_test.append(targets)
    return y_test, y_pred

class Classifier(nn.Module):
    def __init__(self, hidden_dim, freq_dim, attn_heads,dropnum):
        super().__init__()
        self.bert = RobertaClassifier()
        self.fourier_attn = FourierAttention(hidden_dim=hidden_dim, attn_heads=attn_heads)
        self.dropout = nn.Dropout(p=dropnum)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim+freq_dim, 256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Sigmoid()
        )
        self.aigc_head = nn.Sequential(
            nn.Dropout(dropnum),
            nn.Linear(hidden_dim+freq_dim, 256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        self.selfattn = nn.MultiheadAttention(hidden_dim, num_heads= attn_heads, batch_first=True)
        self.freq_lowdim = nn.Linear(hidden_dim, freq_dim)
        self.seq_low_dim = nn.Linear(hidden_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.hidden_dim = hidden_dim
    def forward(self, input_ids, attention_masks):
        '''文本图结构特征提取'''
        seq_feat = self.bert(input_ids = input_ids,attention_mask = attention_masks)
        freq_feat = self.fourier_attn(seq_feat.last_hidden_state)
        gate = self.gate(torch.cat([freq_feat, seq_feat[1]], dim=-1))
        weight_freq_feat = self.freq_lowdim(gate * freq_feat)
        gated_output = torch.cat([weight_freq_feat, (1-gate) * seq_feat[1]], dim=-1)
        #pooled_output = self.pool(gated_output.permute(0, 2, 1)).squeeze()
        logit = self.out_layer(self.dropout(gated_output)) # seq_feat[1]

        aigc_logits = self.aigc_head(gated_output)
        return logit, aigc_logits #prob 


    
class Trainset(Dataset):
    def __init__(self, input_ids, masks,
                 restyle_input_ids_1, restyle_masks_1,
                 restyle_input_ids_2, restyle_masks_2,
                 label, max_len):
        self.input_ids = input_ids
        self.masks = masks
        self.restyle_input_ids_1 = restyle_input_ids_1
        self.restyle_masks_1 = restyle_masks_1
        self.restyle_input_ids_2 = restyle_input_ids_2
        self.restyle_masks_2 = restyle_masks_2

        self.label  = label
        self.max_len  = max_len
    
    def __getitem__(self, item):
        return {
            'input_ids': self.input_ids[item],
            'input_ids_1': self.restyle_input_ids_1[item],
            'input_ids_2': self.restyle_input_ids_2[item],
            'attention_mask':self.masks[item],
            'attention_mask_1':self.restyle_masks_1[item],
            'attention_mask_2':self.restyle_masks_2[item],
            'label': torch.tensor(self.label[item], dtype=torch.long),
            'idx': item
        }
    def __len__(self):
        return self.input_ids.size(0)

class Testset(Dataset):
    def __init__(self, input_ids, masks, label, max_len):
        self.input_ids = input_ids
        self.masks = masks
        self.label = label
        self.max_len = max_len
    
    def __getitem__(self, item):

        return {
            'input_ids': self.input_ids[item],
            'attention_mask': self.masks[item],
            'label': torch.tensor(self.label[item], dtype=torch.long),
            'idx': item
        }
    def __len__(self):
        return self.input_ids.size(0)

    
def create_eval_loader(input_ids, masks, label, max_len, batch_size):
    ds = Testset(input_ids, masks, np.array(label), max_len) 
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

def triplet_loss(aigc_prob, aigc_obj_prob, aigc_emo_prob, margin=1.0):
    # 计算锚点（aigc_prob）与正样本（aigc_obj_prob 和 aigc_emo_prob）的距离
    dist_pos1 = F.pairwise_distance(aigc_prob, aigc_obj_prob, p=2)
    dist_pos2 = F.pairwise_distance(aigc_prob, aigc_emo_prob, p=2)
    
    dist_pos_pair = F.pairwise_distance(aigc_obj_prob, aigc_emo_prob, p=2)
    
    loss = torch.relu(dist_pos1 - dist_pos_pair + margin) + \
           torch.relu(dist_pos2 - dist_pos_pair + margin)
    
    return loss.mean()

def train2test():
    #调用训练数据、测试数据、改写风格的测试数据
    train_input_ids, train_masks, label_train = load_origindata_train(args.dataset_name)
    #调用改写风格的训练数据
    restyle_input_ids_train1, restyle_masks_train1, restyle_input_ids_train2, restyle_masks_train2 = load_reframingsdata(args.dataset_name)
    
    #将原训练数据、改写风格的训练数据整合为训练集格式
    trainset = Trainset(input_ids = train_input_ids,  masks = train_masks,
                        restyle_input_ids_1=restyle_input_ids_train1, restyle_masks_1=restyle_masks_train1,
                        restyle_input_ids_2=restyle_input_ids_train2, restyle_masks_2=restyle_masks_train2,
                        label=label_train, max_len=args.max_len)

    #加载训练集
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=5)
    #加载测试集testloader、改写风格的测试集testloader_res
    test_input_ids, test_masks, test_label = load_origindata_test(args.dataset_name)
    test_loader  = create_eval_loader(input_ids = test_input_ids['O'], masks = test_masks['O'], 
                                label=test_label, max_len=args.max_len, batch_size = args.batch_size)
    test_loader_res_A  = create_eval_loader(input_ids = test_input_ids['A'], masks = test_masks['A'], 
                                label=test_label, max_len=args.max_len, batch_size = args.batch_size)
    test_loader_res_B  = create_eval_loader(input_ids = test_input_ids['B'], masks = test_masks['B'], 
                                label=test_label, max_len=args.max_len,batch_size = args.batch_size)
    test_loader_res_C  = create_eval_loader(input_ids = test_input_ids['C'], masks = test_masks['C'], 
                                label=test_label, max_len=args.max_len,batch_size = args.batch_size)
    test_loader_res_D  = create_eval_loader(input_ids = test_input_ids['D'], masks = test_masks['D'], 
                                label=test_label, max_len=args.max_len,batch_size = args.batch_size)

    model = Classifier(args.hidden_dim, args.freq_dim, args.attn_heads, args.dropout_num).to(device)
    train_losses = []
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    CEloss = nn.CrossEntropyLoss()
    KLloss = nn.KLDivLoss(reduction = 'batchmean')
    # 实验开始记录
    if iter == 0:
        logging.info("\n" + "="*80)
        logging.info(f"EXPERIMENT STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total Args: {args}")
        logging.info("="*80 + "\n")
    logging.info(f"\n{' ITERATION ' + str(iter) + ' ':=^80}")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        
        for batch in tqdm(trainloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)


            
            
            label = batch['label'].to(device)
            
            logit, aigc_logit = model(input_ids = input_ids, attention_masks=attention_mask)
            logit_obj, aigc_obj_logit = model(input_ids = input_ids_1, attention_masks=attention_mask_1) 
            logit_emo, aigc_emo_logit = model(input_ids = input_ids_2, attention_masks=attention_mask_2) 
            
            prob = F.softmax(logit, dim = -1)

            aigc_prob = F.softmax(aigc_logit, dim=-1)
            aigc_obj_prob = F.softmax(aigc_obj_logit, dim=-1)
            aigc_emo_prob = F.softmax(aigc_emo_logit, dim=-1)

            aigcd_loss = triplet_loss(aigc_prob, aigc_obj_prob, aigc_emo_prob, margin=1.0)
            #aigcd_loss = CEloss(aigc_prob, zero_label) + CEloss(aigc_obj_prob, one_label)+ CEloss(aigc_emo_prob, one_label)


            prob_obj_log = F.log_softmax(logit_obj, dim = -1)
            prob_emo_log = F.log_softmax(logit_emo, dim = -1)

            sup_loss = CEloss(logit, label)
            cons_loss= 0.5*KLloss(prob_obj_log, prob) + 0.5*KLloss(prob_emo_log, prob)
            loss = sup_loss + cons_loss + args.loss_weight*aigcd_loss
    
            optimizer.zero_grad() # 清空梯度
            loss.backward() #计算本epoch 的梯度
            epoch_loss.append(loss.item())
            optimizer.step() # 优化模型参数
            scheduler.step() # 优化学习率
        train_losses.append(np.mean(epoch_loss))
        torch.save(model.state_dict(), 'checkpoints/ERFD/' + datasetname + '_iter' + str(iter) + '.m')
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, np.mean(epoch_loss)))
        if epoch == args.epochs - 1:
            model.eval()
            y_pred = []
            y_test = []
            y_pred_res_A = []
            y_pred_res_B = []
            y_pred_res_C = []
            y_pred_res_D = []

            y_test, y_pred = test_ouput(test_loader, model)
            _, y_pred_res_A = test_ouput(test_loader_res_A, model)
            _, y_pred_res_B = test_ouput(test_loader_res_B, model)
            _, y_pred_res_C = test_ouput(test_loader_res_C, model)
            _, y_pred_res_D = test_ouput(test_loader_res_D, model)

            combined_pred =y_pred_res_A + y_pred_res_B + y_pred_res_C + y_pred_res_D
            combined_true = y_test+y_test+y_test+y_test
            orig_acc, orig_prec, orig_rec, orig_f1 = output_metrics_metrics(y_test, y_pred, 'Origin')
            A_acc, A_prec, A_rec, A_f1 = output_metrics_metrics(y_test, y_pred_res_A, 'A')
            B_acc, B_prec, B_rec, B_f1 = output_metrics_metrics(y_test, y_pred_res_B, 'B')
            C_acc, C_prec, C_rec, C_f1 = output_metrics_metrics(y_test, y_pred_res_C, 'C')
            D_acc, D_prec, D_rec, D_f1 = output_metrics_metrics(y_test, y_pred_res_D, 'D')
            Comb_acc, Comb_prec, Comb_rec, Comb_f1 = output_metrics_metrics(combined_true, combined_pred, 'Combined')
            # 存储本次迭代结果
            iteration_results = {
                'original': (orig_acc, orig_prec, orig_rec, orig_f1),
                'A': (A_acc, A_prec, A_rec, A_f1),
                'B': (B_acc, B_prec, B_rec, B_f1),
                'C': (C_acc, C_prec, C_rec, C_f1),
                'D': (D_acc, D_prec, D_rec, D_f1),
                'combined': (Comb_acc, Comb_prec, Comb_rec, Comb_f1)
            }

            return iteration_results
if __name__ == "__main__":

    # 配置日志记录
    log_filename = os.path.join('logs/para/para_rest', f'{args.dataset_name}_{args.model_name}_{args.loss_weight}_{args.freq_dim}_{args.attn_heads}_{args.epochs}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    datasetname=args.dataset_name
    batch_size = args.batch_size
    max_len = 512
    n_epochs = args.epochs
    iterations=args.iters

    all_results = []


    for iter in range(iterations):
        set_seed(iter)
        results = train2test()
        all_results.append(results)
    # 最终统计分析
    logging.info("\n" + "="*80)
    logging.info("FINAL STATISTICAL ANALYSIS (Ten Run AVERAGED)")
    logging.info("="*80)
    versions = ['original', 'A', 'B', 'C', 'D', 'combined']
    for version in versions:
        accs = [result[version][0] for result in all_results]
        precs = [result[version][1] for result in all_results]
        recs = [result[version][2] for result in all_results]
        f1s = [result[version][3] for result in all_results]
        
        logging.info(f"\n{version.upper()} SUMMARY ({len(accs)} iterations):")
        logging.info(f"Macro Accuracy:  {np.mean(accs):.2f}% ± {np.std(accs, ddof=1):.2f}%")
        logging.info(f"Macro Precision: {np.mean(precs):.2f}% ± {np.std(precs, ddof=1):.2f}%")
        logging.info(f"Macro Recall:    {np.mean(recs):.2f}% ± {np.std(recs, ddof=1):.2f}%")
        logging.info(f"Macro F1:        {np.mean(f1s):.2f}% ± {np.std(f1s, ddof=1):.2f}%")
    
    # 显著性检验（macro F1比较）