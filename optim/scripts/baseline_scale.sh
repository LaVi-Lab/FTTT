set -e

OUTPUT_DIR=$1
DEVICE=${2:-0}
SEED=${3:-42}

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=${DEVICE}

export RM_API=http://175.102.19.159:6003
export OJ_API=http://localhost:9999

RED='\033[0;31m'
NC='\033[0m'

if [[ "${OUTPUT_DIR}" == *"MATH500"* ]]; then
    DATASET=MATH500
elif [[ "${OUTPUT_DIR}" == *"GSM8K"* ]]; then
    DATASET=GSM8K
elif [[ "${OUTPUT_DIR}" == *"MBPP"* ]]; then
    DATASET=MBPP
else
    echo "${RED}Cannot parse dataset from ${OUTPUT_DIR}${NC}"
    exit
fi

if [[ "${OUTPUT_DIR}" == *"Llama-3.1-8B-Instruct"* ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/llama_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ "${OUTPUT_DIR}" == *"Mistral-7B-Instruct-v0.3"* ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/mistral_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
else
    echo "${RED}Cannot parse base model name from ${OUTPUT_DIR}${NC}"
    exit
fi

python ${SCRIPT_DIR}/../baseline_eval.py \
    --dataset_name ${DATASET} \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --seed ${SEED} \
    --per_device_eval_batch_size 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --num_return_sequences 8 \
    --num_repeat 4 \
    --max_new_tokens 2048 \
    --output_dir ${OUTPUT_DIR} \
    --skip_cases_path ${SKIP_PATH} 2>&1 | tee -a ${OUTPUT_DIR}/${DATASET}_${SEED}_scale_32.log
