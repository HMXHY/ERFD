import torch.nn as nn
from transformers import RobertaModel

class RobertaClassifier(nn.Module):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        for name, param in self.roberta.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states= True
        )
        return outputs

