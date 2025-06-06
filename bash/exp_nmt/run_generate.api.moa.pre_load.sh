## mode
mode=ensemble_MoA


## I / O params
task=wmt22_test
data_name=test.zh_to_en.num=200.jsonl 
source=$(echo $data_name | cut -d '.' -f 2 | cut -d '_' -f 1)
input=../../data/$task/$data_name
mkdir -p ../../output/$task/
save_mode='a'


## model params
model_num=5
root_configs=../launch_large_models_sglang/server_configs_ensemble/
root_reward_configs=../launch_large_models_sglang/server_reward/
config_name=ensemble_models.reward=reward_model
short_config_name=$config_name


## generation params
batch_size=1000
parallel_num=500

n_iter=3
num_aggregation=${model_num}

for n_samples in $1 ; do
## sampling params
max_tokens=2048
temperature=0.7
top_p=1

path_to_translation_template=../../prompts/translation.txt
path_to_refine_template=../../prompts/aggregator.nmt.v2.txt

output=../../output/$task/${data_name}.source=${source}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.n_iter=${n_iter}.num_agg=${num_aggregation}.temp=${temperature}.top_p=${top_p}.agg=v2.jsonl

python ../../code/ensemble_inference.nmt.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --root_reward_configs $root_reward_configs \
                                     --n_samples $n_samples \
                                     --path_to_translation_template $path_to_translation_template \
                                     --path_to_refine_template $path_to_refine_template \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size \
                                     --n_iter $n_iter \
                                     --num_aggregation $num_aggregation \
                                     --source $source

done;
                                     
