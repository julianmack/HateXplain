import argparse
from collections import Counter,defaultdict
import json

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm

from hateXplain.Preprocess.dataCollect import load_dataset_df

# The bias methods that will be considered
method_list = ['subgroup', 'bpsn', 'bnsp']


def eval_bias(subset='test', explanations_fp=None, explanations_dict=None):
    df_bias, community_list = get_df_bias(subset)

    metrics = bias_metric_not_agg(
        explanations_fp=explanations_fp,
        explanations_dict=explanations_dict,
        df_bias=df_bias,
        community_list=community_list
    )

    results = gen_final_bias_metrics(
        metrics,
        num_communities=len(community_list)
    )
    print(f'{explanations_fp}, SUBSET={subset}')
    for k, v in results.items():
        print(f'\t{k.upper()}: {v:.6f}')
    return results


def get_df_bias(subset):
    df = load_dataset_df(num_classes=2)
    target_info, all_communities_selected = generate_target_info(df)

    community_count_dict = Counter(all_communities_selected)

    # We remove None and Other from dictionary
    community_count_dict.pop('None')
    community_count_dict.pop('Other')

    # For the bias calculation, we are considering the top 10 communites based on their count
    list_selected_community = [community for community, value in community_count_dict.most_common(10)]
    community_list = list(list_selected_community)

    final_target_info ={}
    for each in target_info:
        temp = list(set(target_info[each])&set(list_selected_community))
        if len(temp) == 0:
            final_target_info[each] = None
        else:
            final_target_info[each] = temp

    # Add a new column 'final_target_category' which will contain the selected target community names
    df['final_target_category'] = df['post_id'].map(final_target_info)

    # The post_id_divisions file stores the train, val, test split ids. We select only the test ids.
    postpost_id_divisions_path = './Data/post_id_divisions.json'

    with open(postpost_id_divisions_path, 'r') as fp:
        post_id_dict=json.load(fp)

    df_bias = df[df['post_id'].isin(post_id_dict[subset])]

    return df_bias, community_list


def generate_target_info(dataset):
    """Extracts group target information.txt

    This function is used to extract the target community based on majority
    voting. If at least 2 annoatators (out of 3) have selected a community,
    then we consider it.
    """
    final_target_output = defaultdict(list)
    all_communities_selected = []

    for each in dataset.iterrows():
        # All the target communities tagged for this post
        all_targets = each[1]['target1']+each[1]['target2']+each[1]['target3']
        community_dict = dict(Counter(all_targets))

        # Select only those communities which are present more than once.
        for key in community_dict:
            if community_dict[key]>1:
                final_target_output[each[1]['post_id']].append(key)
                all_communities_selected.append(key)

        # If no community is selected based on majority voting then we don't select any community
        if each[1]['post_id'] not in final_target_output:
            final_target_output[each[1]['post_id']].append('None')
            all_communities_selected.append(key)

    return final_target_output, all_communities_selected


def bias_evaluation_metric(dataset, method, community):
    """Divides ids into postive or negative class based on the method."""
    positive_ids = []
    negative_ids = []
    if method=='subgroup':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] =='non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    positive_ids.append(eachrow[1]['post_id'])
            else:
                pass
    elif method=='bpsn':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] =='non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    pass
            else:
                if eachrow[1]['final_label'] !='non-toxic':
                    positive_ids.append(eachrow[1]['post_id'])
                else:
                    pass
    elif method=='bnsp':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] !='non-toxic':
                    positive_ids.append(eachrow[1]['post_id'])
                else:
                    pass
            else:
                if eachrow[1]['final_label'] =='non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    pass
    else:
        print('Incorrect option selected!!!')

    return {'positiveID':positive_ids, 'negativeID':negative_ids}

def bias_metric_not_agg(
    df_bias, community_list, explanations_fp=None, explanations_dict=None,
):
    assert explanations_dict or explanations_fp
    assert not (explanations_dict and explanations_fp)
    total_data ={}
    with open(explanations_fp) as fp:
        for line in fp:
            data = json.loads(line)
            total_data[data['annotation_id']] = data
    results = {k: {} for k in method_list}
    for each_method in method_list:
        for each_community in community_list:
            community_data = bias_evaluation_metric(df_bias, each_method, each_community)
            truth_values = []
            prediction_values = []


            label_to_value = {'toxic':1.0, 'non-toxic':0.0}
            for each in community_data['positiveID']:
                truth_values.append(label_to_value[total_data[each]['ground_truth']])
                prediction_values.append(convert_to_score(total_data[each]['classification'], total_data[each]['classification_scores']))

            for each in community_data['negativeID']:
                truth_values.append(label_to_value[total_data[each]['ground_truth']])
                prediction_values.append(convert_to_score(total_data[each]['classification'], total_data[each]['classification_scores']))

            roc_output_value = roc_auc_score(truth_values, prediction_values)
            results[each_method][each_community] = roc_output_value
    return results

def gen_final_bias_metrics(results_dict, num_communities, power_value = -5):
    output = {}
    for each_method in results_dict:
        temp_value =[]
        for each_community in results_dict[each_method]:
            temp_value.append(pow(results_dict[each_method][each_community], power_value))
        value = pow(np.sum(temp_value)/num_communities, 1/power_value)
        output[each_method] = value
    return output

def convert_to_score(label_name, label_dict):
    """Converts the classification into a [0-1] score.

    A value of 0 meaning non-toxic and 1 meaning toxic.
    """
    if label_name=='non-toxic':
        return 1-label_dict[label_name]
    else:
        return label_dict[label_name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'explanation_dict_name',
        type=str,
        help='explanation dict name to use for evaluation. '
            'eg `bestModel_bert_base_uncased_Attn_train_TRUE_bias.json`'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='val',
        help='data subset'
    )


    args = parser.parse_args()
    explanations_fp = f'./explanations_dicts/{args.explanation_dict_name}'
    eval_bias(explanations_fp=explanations_fp, subset=args.subset)
