set -e

export CUDA_VISIBLE_DEVICES=${5:-0}
DATASET=$1
SEED=$2
MODEL=$3
PEFT=$4

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [[ "${MODEL}" == "Llama-3.1-8B-Instruct" ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/llama_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ "${MODEL}" == "Mistral-7B-Instruct-v0.3" ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/mistral_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
else
    echo "${RED}Do not supprt ${MODEL}${NC}"
    exit
fi

if [[ "${PEFT}" == "LoRA" ]]; then
    # Stable
    MODEL_ARGS="--peft_type LoRA \
--lora_r 16 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_layers_to_transform 24 25 26 27 28 29 30 31 \
--learning_rate 2e-5 \
--fp16 \
--max_length 4096"
elif [[ "${PEFT}" == "FT" ]]; then
    # Stable
    MODEL_ARGS="--peft_type ${PEFT} \
--fp16 \
--learning_rate 1e-5 \
--max_length 1024"
elif [[ "${PEFT}" == "Prefix" ]]; then
    # Stable
    if [[ "${MODEL}" == "Llama-3.1-8B-Instruct" ]]; then
        LR=2e-4
    elif [[ "${MODEL}" == "Mistral-7B-Instruct-v0.3" ]]; then
        LR=5e-5
    else
        echo "${RED}Do not know the learning rate for ${MODEL}${NC}"
        exit
    fi
    MODEL_ARGS="--peft_type Prefix \
--prefix_length 10 \
--learning_rate ${LR} \
--fp16 \
--max_length 4096"
elif [[ "${PEFT}" == "Adapter" ]]; then
    # Stable
    MODEL_ARGS="--peft_type Adapter \
--reduction_factor 16 \
--learning_rate 1e-4 \
--fp16 \
--max_length 4096"
#     LEAVEOUT_LAYERS=$(for i in {0..15}; do echo -n "${i},"; done)
# --leave_out ${LEAVEOUT_LAYERS::-1} \
elif [[ "${PEFT}" == "IA3" ]]; then
    # Stable
    MODEL_ARGS="--peft_type IA3 \
--learning_rate 5e-5 \
--fp16 \
--max_length 4096"
elif [[ "${PEFT}" == "LNTuning" ]]; then
    # Stable
    MODEL_ARGS="--peft_type LNTuning \
--learning_rate 4e-4 \
--fp16 \
--max_length 4096"
elif [[ "${PEFT}" == "ReFT" ]]; then
    MODEL_ARGS="--peft_type ReFT \
--reft_layers all \
--reft_r 1 \
--learning_rate 2e-5 \
--fp16 \
--max_length 4096"
# --reft_prefix_positions 3 \
# --reft_suffix_positions 0 \
else
    echo "${RED}Do not supprt ${PEFT}${NC}"
    exit
fi

if [[ "${DATASET}" == "MBPP" ]]; then
    TRAIN_ARGS="--num_train_epochs 100 \
--gradient_accumulation_steps 20 \
--save_every_n_epochs 10"
elif [[ "${DATASET}" == "GSM8K" ]]; then
    TRAIN_ARGS="--num_train_epochs 10 \
--gradient_accumulation_steps 20 \
--save_every_n_epochs 1"
elif [[ "${DATASET}" == "MATH500" ]]; then
    TRAIN_ARGS="--num_train_epochs 3 \
--gradient_accumulation_steps 20 \
--save_every_n_epochs 1"
else
    echo "${RED}Do not supprt ${DATASET}${NC}"
    exit
fi

export RM_API=http://175.102.19.159:6003
export OJ_API=http://localhost:9999

RED='\033[0;31m'
NC='\033[0m'

OUTPUT_DIR=outputs/${MODEL}_${DATASET}_seed${SEED}_${PEFT}

mkdir -p ${OUTPUT_DIR}
cp "$0" ${OUTPUT_DIR}/

python ${SCRIPT_DIR}/../baseline.py \
    --seed ${SEED} \
    --model_name_or_path ${BASE_MODEL_PATH}  \
    ${MODEL_ARGS} \
    ${TRAIN_ARGS} \
    --dataset_name ${DATASET} \
    --skip_cases_path ${SKIP_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" | tee -a ${OUTPUT_DIR}/train.log
