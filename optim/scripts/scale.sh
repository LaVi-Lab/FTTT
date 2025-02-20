set -e

OUTPUT_DIR=$1
DEVICES=${2:-0}

# INPUT_STRATEGY=${3:-restart}
# OUTPUT_STRATEGY=${4:-current}

SEED=${3:-42}
LOCAL_BATCH_SIZE=1

export RM_API=http://175.102.19.159:6003
export OJ_API=http://localhost:9999

# Automatically set up everything
export CUDA_VISIBLE_DEVICES=${DEVICES}

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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [[ "${OUTPUT_DIR}" == *"Llama-3.1-8B-Instruct"* ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/llama_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ "${OUTPUT_DIR}" == *"Mistral-7B-Instruct-v0.3"* ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/mistral_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
else
    echo "${RED}Cannot parse base model name ${OUTPUT_DIR}${NC}"
    exit
fi

# Run!
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
python ${SCRIPT_DIR}/../run.py eval \
    --output_dir ${OUTPUT_DIR} \
    --dataset ${DATASET} \
    --model ${BASE_MODEL_PATH} \
    --seed ${SEED} \
    --feedback_type gt \
    --max_new_tokens 2048 \
    --num_trials 32 \
    --skip_cases_path ${SKIP_PATH} \
    --batch_size ${LOCAL_BATCH_SIZE} 2>&1 | tee -a ${OUTPUT_DIR}/${DATASET}_${SEED}_scale.log
