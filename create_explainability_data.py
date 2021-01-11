import argparse
import json
from pathlib import Path
import os
import ast

import more_itertools as mit
from transformers import BertTokenizerFast

from Preprocess.dataCollect import load_dataset_df
from Preprocess.spanMatcher import returnMask

def create_explainability_data(bert_model=True):
    if bert_model:
        params = {
            'path-files': 'bert-base-uncased',
            'bert_tokens': True,
            'max_length': 128,
            'include_special': False
        }
        save_path = './Data/Evaluation/Model_Eval/bert/'
        tokenizer = BertTokenizerFast.from_pretrained(
            params['path-files'],
            do_lower_case=False
        )
    else:
        params = {
            'bert_tokens': False,
            'max_length': 128,
            'include_special': False
        }
        save_path = './Data/Evaluation/Model_Eval/None/'
        tokenizer = None

    with open('./Data/post_id_divisions.json') as fp:
        id_division = json.load(fp)

    method = 'union'
    save_split = True
    convert_to_eraser_format(
        dataset=get_training_data(params, tokenizer=tokenizer),
        method='union',
        save_split=True,
        save_path=save_path,
        id_division=id_division,
    )


def get_training_data(params, tokenizer):
    """Loads dataset and gets the tokenwise rationales."""
    df = load_dataset_df(num_classes=3)
    post_ids_list=[]
    text_list=[]
    attention_list=[]
    label_list=[]

    final_binny_output = []
    for index, row in df.iterrows():
        annotation=row['final_label']

        text=row['text']
        post_id=row['post_id']
        annotation_list=[row['label1'],row['label2'],row['label3']]
        tokens_all = list(row['text'])

        if(annotation!= 'undecided'):
            tokens_all, attention_masks = returnMask(row, params, tokenizer)
            final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

    return final_binny_output


# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]

# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each==1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each)== int:
            start = each
            end = each+1
        elif len(each) == 2:
            start = each[0]
            end = each[1]+1
        else:
            print('error')

        output.append({"docid":post_id,
              "end_sentence": -1,
              "end_token": end,
              "start_sentence": -1,
              "start_token": start,
              "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output

# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):
    final_output = []
    Path(save_path).mkdir(parents=True, exist_ok=True)
    if save_split:
        train_fp = open(save_path+'train.jsonl', 'w')
        val_fp = open(save_path+'val.jsonl', 'w')
        test_fp = open(save_path+'test.jsonl', 'w')

    for tcount, eachrow in enumerate(dataset):

        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]

        if majority_label=='normal':
            continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]


        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)

        if save_split:
            if not os.path.exists(save_path+'docs'):
                os.makedirs(save_path+'docs')

            with open(save_path+'docs/'+post_id, 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))

            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp)+'\n')

            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp)+'\n')

            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp)+'\n')
            else:
                print(post_id)

    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()

    return final_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'bert_model',
        type=ast.literal_eval,
        help='Boolean. True if bert uncased model.'
    )

    args = parser.parse_args()
    create_explainability_data(
        bert_model=args.bert_model,
    )
