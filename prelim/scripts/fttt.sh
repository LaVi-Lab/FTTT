set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_VERBOSITY=error

export OJ_API=http://localhost:9999

RED='\033[0;31m'
NC='\033[0m'

MODEL=$1
DATASET=$2
SEED=$3
MODE=${4:-TTT}

# SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [[ "${MODEL}" == "Llama-3.1-8B-Instruct" ]]; then
    MODEL_DIR=meta-llama/Llama-3.1-8B-Instruct
    ARCH=llama
elif [[ "${MODEL}" == "Mistral-7B-Instruct-v0.3" ]]; then
    MODEL_DIR=mistralai/Mistral-7B-Instruct-v0.3
    ARCH=mistral
else
    echo "${RED}Do not supprt ${MODEL}${NC}"
    exit
fi

declare -A model_task_lrs

model_task_lrs["Llama-3.1-8B-Instruct:MATH500"]=1e-5
model_task_lrs["Llama-3.1-8B-Instruct:GSM8K"]=1e-5
model_task_lrs["Llama-3.1-8B-Instruct:MBPP"]=1e-5
model_task_lrs["Llama-3.1-8B-Instruct:HumanEval"]=1e-5
model_task_lrs["Mistral-7B-Instruct-v0.3:MATH500"]=1e-5
model_task_lrs["Mistral-7B-Instruct-v0.3:GSM8K"]=1e-5
model_task_lrs["Mistral-7B-Instruct-v0.3:MBPP"]=1e-5
model_task_lrs["Mistral-7B-Instruct-v0.3:HumanEval"]=1e-5

if [[ "${MODE}" == "FTTT" ]]; then
    DEFAULT_VERSION=fttt
elif [[ "${MODE}" == "FTTT+" ]]; then
    DEFAULT_VERSION=self_reflect_norm
    model_task_lrs["Mistral-7B-Instruct-v0.3:MBPP"]=2e-5
    model_task_lrs["Mistral-7B-Instruct-v0.3:HumanEval"]=2e-5
else
    echo "${RED}Do not support ${MODE}${NC}"
    exit
fi

key="${MODEL}:${DATASET}"
DEFAULT_LR=${model_task_lrs[$key]}
LR=${5:-$DEFAULT_LR}
VERSION=${6:-$DEFAULT_VERSION}

MODEL_NAME=${MODEL}-${DATASET}-seed-${SEED}-${VERSION}
OUTPUT_DIR=outputs/${MODEL_NAME}

mkdir -p ${OUTPUT_DIR}
cp "$0" ${OUTPUT_DIR}/

echo "dataset: ${DATASET}"$'\n'"seed: ${SEED}"$'\n'"version: ${VERSION}"$'\n'"lr: ${LR}"$'\n'"path: ${OUTPUT_DIR}" | tee ${OUTPUT_DIR}/output.log

python prelim/fttt.py \
    --model_name_or_path ${MODEL_DIR} \
    --model_arch ${ARCH} \
    --dataset ${DATASET} \
    --seed ${SEED} \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --run_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_trials 32 \
    --num_return_sequences 1 \
    --fp16 True \
    --per_device_eval_batch_size 1 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 0 \
    --learning_rate ${LR} \
    --num_optim_step_per_sample 1 \
    --method ${VERSION} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --q_lora False 2>&1 | tee -a ${OUTPUT_DIR}/output.log
