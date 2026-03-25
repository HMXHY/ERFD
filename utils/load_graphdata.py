import pickle
import numpy as np
def load_origindata_train(obj):
    train_data = pickle.load(open('data_ids/news_articles/' + obj + '_train.pkl', 'rb'))

    label_train = train_data['labels']
    train_input_ids = train_data['input_ids']
    train_masks = train_data['attention_masks']

    
    print(obj, 'origin train set load')
    return train_input_ids, train_masks, label_train

def load_origindata_test(obj):
    test_dict  = pickle.load(open('data_ids/news_articles/' + obj + '_test.pkl', 'rb'))
    test_label = test_dict['labels']

    test_input_ids = {}
    test_masks = {}
    for testtype in ['O','A','B','C','D']:
        if testtype=='O':
            test_input_ids_masks  = pickle.load(open('data_ids/news_articles/' + obj + '_test.pkl', 'rb'))
            test_input_ids[testtype] = test_input_ids_masks['input_ids']
            test_masks[testtype] = test_input_ids_masks['attention_masks']
        else:
            test_input_ids_masks = pickle.load(open('data_ids/adversarial_test/'+obj+'_test_adv_'+ testtype +'.pkl','rb'))
            test_input_ids[testtype] = test_input_ids_masks['input_ids']
            test_masks[testtype] = test_input_ids_masks['attention_masks']
    print(obj, 'test set load')
    return test_input_ids, test_masks, test_label

def load_reframingsdata(obj):
    print("loading news reframings")
    print("Dataset: ", obj)

    restyle_text_train1_1 = pickle.load(open('data_ids/reframings/' + obj+ '_train_objective.pkl', 'rb'))
    restyle_text_train1_2 = pickle.load(open('data_ids/reframings/' + obj+ '_train_neutral.pkl', 'rb'))
    restyle_text_train2_1 = pickle.load(open('data_ids/reframings/' + obj+ '_train_emotionally_triggering.pkl', 'rb'))
    restyle_text_train2_2 = pickle.load(open('data_ids/reframings/' + obj+ '_train_sensational.pkl', 'rb'))

    restyle_input_ids_train1 = np.array(restyle_text_train1_1['input_ids'])
    restyle_input_ids_train1_2 = np.array(restyle_text_train1_2['input_ids'])
    restyle_input_ids_train2 = np.array(restyle_text_train2_1['input_ids'])
    restyle_input_ids_train2_2 = np.array(restyle_text_train2_2['input_ids'])

    restyle_masks_train1 = np.array(restyle_text_train1_1['attention_masks'])
    restyle_masks_train1_2 = np.array(restyle_text_train1_2['attention_masks'])
    restyle_masks_train2 = np.array(restyle_text_train2_1['attention_masks'])
    restyle_masks_train2_2 = np.array(restyle_text_train2_2['attention_masks'])


    replace_idx = np.random.choice(len(restyle_input_ids_train1), len(restyle_input_ids_train1) // 2, replace=False)

    restyle_input_ids_train1[replace_idx] = restyle_input_ids_train1_2[replace_idx]
    restyle_input_ids_train2[replace_idx] = restyle_input_ids_train2_2[replace_idx]
    restyle_masks_train1[replace_idx] = restyle_masks_train1_2[replace_idx]
    restyle_masks_train2[replace_idx] = restyle_masks_train2_2[replace_idx]

    return restyle_input_ids_train1, restyle_masks_train1, restyle_input_ids_train2, restyle_masks_train2
    
    