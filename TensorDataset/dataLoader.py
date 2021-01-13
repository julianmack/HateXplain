import torch
import transformers
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder


def custom_att_masks(input_ids):
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks

def combine_features(tuple_data,params,is_train=False, batch_size=None):
    input_ids =  [ele[0] for ele in tuple_data]
    att_vals = [ele[1] for ele in tuple_data]
    labels = [ele [2] for ele in tuple_data]


    encoder = LabelEncoder()

    encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
    labels=encoder.transform(labels)

    dataloader=return_dataloader(
        input_ids=input_ids,
        labels=labels,
        att_vals=att_vals,
        batch_size=batch_size or params['batch_size'],
        params=params,
        is_train=is_train
    )
    return dataloader

def return_dataloader(input_ids,labels,att_vals,batch_size,params,is_train=False):
    data = BasicDataset(input_ids,att_vals,labels)
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(
        data,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=get_collate_fn(params),
        num_workers=3,
    )
    return dataloader

def get_collate_fn(params):

    def collate_fn(batch):
        input_ids, attentions, labels = [], [], []
        for item in batch:
            a, b, c = item

            input_ids.append(a)
            attentions.append(b)
            labels.append(c)

        input_ids = pad_sequences(input_ids, dtype="long",
                            value=0, truncating="post", padding="post")
        attentions = pad_sequences(attentions, dtype="float",
                            value=0.0, truncating="post", padding="post")

        masks = torch.tensor(np.array(custom_att_masks(input_ids)), dtype=torch.bool)
        labels = torch.tensor(labels, dtype=torch.long)

        return torch.tensor(input_ids), torch.tensor(attentions), masks, labels
    return collate_fn

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids,att_vals,labels):
        self.inputs = input_ids
        self.attentions = att_vals
        self.labels = labels

    def __getitem__(self, idx):
        return self.inputs[idx], self.attentions[idx], self.labels[idx]

    def __len__(self):
        return len(self.inputs)
