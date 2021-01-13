from Models.bertModels import *
from Models.otherModels import *

def select_model(params, embeddings):
    if(params['bert_tokens']):
        if(params['what_bert']=='weighted'):
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
