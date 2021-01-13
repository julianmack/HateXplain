#!/usr/bin/env bash
[[ $# -eq 5 && -n "$1" && -n "$2" && -n "$3" && -n "$4" && -n "$5" ]] || {
  >&2 echo \
  "usage: $0 model_name config_name subset attn_lambda num_supervised_heads"
  exit 1
}

model_name="$1"
config_name="$2"
subset="$3"
attn_lambda="$4"
num_supervised_heads="$5"

output_file_rat="metric_summaries/rationale/${config_name}_${attn_lambda}_${num_supervised_heads}_${subset}.json"


python testing_for_bias.py $model_name $attn_lambda --num_supervised_heads $num_supervised_heads --subset $subset
# TODO: save final testing_for_bias results to file


python testing_with_rational.py $model_name $attn_lambda --num_supervised_heads $num_supervised_heads
# NOTE: if you need to create explainability data in eraser format run
# python create_explainability_data.py True/False
cd eraserbenchmark && \
python rationale_benchmark/metrics.py \
    --split $subset --strict --data_dir \
    ../Data/Evaluation/Model_Eval/bert \
    --results "../explanations_dicts/${config_name}_${attn_lambda}_explanation_top5.json" \
    --score_file ../$output_file_rat
