#!/usr/bin/env bash
[[ $# -eq 4 && -n "$1" && -n "$2" && -n "$3" && -n "$4" ]] || {
  >&2 echo \
  "usage: $0 model_name config_name subset attn_lambda"
  exit 1
}

model_name="$1"
config_name="$2"
subset="$3"
attn_lambda="$4"

output_file_rat="metric_summaries/rationale/${config_name}_${attn_lambda}_${subset}.json"


# TODO: add --subset $subset to testing_with_rational/testing_for_bias
python testing_for_bias.py $model_name $attn_lambda
python testing_with_rational.py $model_name $attn_lambda


python eval_bias.py "${config_name}_bias.json" --subset $subset
# TODO: save eval_bias result to file

# NOTE: if you need to create explainability data in eraser format run
# python create_explainability_data.py True/False
cd eraserbenchmark && \
python rationale_benchmark/metrics.py \
    --split $subset --strict --data_dir \
    ../Data/Evaluation/Model_Eval/bert \
    --results "../explanations_dicts/${config_name}_${attn_lambda}_explanation_top5.json" \
    --score_file ../$output_file_rat
