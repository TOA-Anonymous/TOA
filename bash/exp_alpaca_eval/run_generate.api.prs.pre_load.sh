## mode
mode=PRS

## I / O params
task=alpaca_eval
data_name=alpaca_eval.num=805.jsonl 
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'

## model params
model_num=1
root_configs=../launch_large_models_sglang/server_configs_mistral-large-instruct-2407/
path_reward_config=../../model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=1.json
config_name=mis-large-2407.reward=ArmoRM
short_config_name=$config_name


## generation params
batch_size=1000  
parallel_num=200 


for n_samples in 160 ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

path_to_refine_template=../../prompts/refinement_wo_feedback.inst_following.v13.txt

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.temp_v=13.jsonl


python ../../code/ensemble_inference.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --path_reward_config $path_reward_config \
                                     --path_to_refine_template $path_to_refine_template \
                                     --n_samples $n_samples \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size

done;
                                     
