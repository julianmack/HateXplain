from typing import List

import numpy as np
from torch import Tensor
from torch.nn import Softmax

from hateXplain.Models.bertModels import *
from hateXplain.Models.otherModels import *

def select_model(params, embeddings):
    if(params['bert_tokens']):
        if(params['what_bert']=='weighted'):
            print(f'Loding pretrained model from {params["path_files"]}')
            model = SC_weighted_BERT.from_pretrained(
                params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
                num_labels = params['num_classes'], # The number of output labels
                output_attentions = True, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
                hidden_dropout_prob=params['dropout_bert'],
                params=params
            )
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        text=params['model_name']
        if(text=="birnn"):
            model=BiRNN(params,embeddings)
        elif(text == "birnnatt"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "birnnscrat"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "cnn_gru"):
            model=CNN_GRU(params,embeddings)
        elif(text == "lstm_bad"):
            model=LSTM_bad(params)
        else:
            print("Error in model name!!!!")
        return model


class LabelMapper():
    """Maps from 3 class labels to two-class labels.

    Specifically, 'non-toxic' in 2-class labelling system maps to 'normal'
    in multi-class labelling system.
    """
    def __init__(self):
        self.classes_2 = list(np.load('Data/classes_two.npy', allow_pickle=True))
        self.classes_n = list(np.load('Data/classes.npy', allow_pickle=True))

        self.normal_idx = self.classes_n.index('normal')
        self.not_normal_idexes = [
            idx for idx in range(len(self.classes_n)) if idx != self.normal_idx
        ]

        self.non_toxic_idx = self.classes_2.index('non-toxic')
        self.toxic_idx = self.classes_2.index('toxic')

        self.softmax = Softmax(dim=1)


    def __call__(self, x: Tensor):
        """Takes (B, n) probs and returns (B, 2) probs.

        The n - 1 class probabilities that are not 'normal' are summed.
        """
        B, _ = x.shape
        output = torch.zeros(B, 2)
        output[:, self.non_toxic_idx] = x[:, self.normal_idx]

        toxic_tensors = [x[:, idx] for idx in self.not_normal_idexes]
        toxic_tensors_stacked = torch.stack(toxic_tensors, dim=1)
        output[:, self.toxic_idx] = torch.sum(toxic_tensors_stacked, dim=1)
        return output


    def multi_class_to_binary_probs(self, pred_probs: Tensor, logits: bool = False):
        """
        Args:
            pred_probs: (B, num_classes).

            logits: bool. If True, logits (instead of probabilities) are passed.

        Returns:
            (B, 2) logits.
        """
        assert len(pred_probs.shape) == 2, f"pred_probs.shape should have 2-dims but {pred_probs.shape=}"
        B, num_classes = pred_probs.shape
        assert num_classes > 2

        if logits:
            pred_probs = self.softmax(pred_probs)

        return self(pred_probs)



    def multi_class_to_binary_labels(self, labels: List[str]):
        raise NotImplementedError()
