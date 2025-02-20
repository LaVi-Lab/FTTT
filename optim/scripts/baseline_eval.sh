set -e

OUTPUT_DIR=$1
DEVICE=${2:-0}
DATASET=${3:-HumanEval}

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=${DEVICE}

export RM_API=http://175.102.19.159:6003
export OJ_API=http://localhost:9999

RED='\033[0;31m'
NC='\033[0m'

if [[ "${OUTPUT_DIR}" == *"Llama-3.1-8B-Instruct"* ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/llama_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ "${OUTPUT_DIR}" == *"Mistral-7B-Instruct-v0.3"* ]]; then
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/mistral_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
else
    echo "${RED}Cannot parse ${OUTPUT_DIR}${NC}"
    exit
fi

python ${SCRIPT_DIR}/../baseline_eval.py \
    --dataset_name ${DATASET} \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --skip_cases_path ${SKIP_PATH} 2>&1 | tee -a ${OUTPUT_DIR}/${DATASET}_eval.log
