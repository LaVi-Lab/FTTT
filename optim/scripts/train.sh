set -e

GPUS=(0)

DATASET=$1
MODEL=$3
SEED=$2
GLOBAL_BATCH_SIZE=20
LOCAL_BATCH_SIZE=1  # We cannot use local_batch_size > 1 as we average predicted updates across samples within a batch

OUTPUT_DIR=outputs/${MODEL}_${DATASET}_seed${SEED}_our

export RM_API=http://175.102.19.159:6003
export OJ_API=http://localhost:9999
export CUDA_LAUNCH_BLOCKING=1

# Automatically set up everything
NGPU=${#GPUS[@]}

export CUDA_VISIBLE_DEVICES=$( IFS=$','; echo "${GPUS[*]}" )

RED='\033[0;31m'
NC='\033[0m'

if [[ "${MODEL}" == "Llama-3.1-8B-Instruct" || "${MODEL}" == "Mistral-7B-Instruct-v0.3" ]]; then
    INNER_PARAMS=$(for i in {30..31}; do echo -n "model.layers.${i}.self_attn.q_proj.weight model.layers.${i}.self_attn.v_proj.weight "; done)
else
    echo "${RED}Do not supprt ${MODEL}${NC}"
    exit
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
if [[ "${MODEL}" == "Llama-3.1-8B-Instruct" ]]; then
    # ARCH=Llama
    FEEDBACK="<|start_header_id|>user<|end_header_id|>\n\nYour answer is incorrect.<|eot_id|>"
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/llama_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ "${MODEL}" == "Mistral-7B-Instruct-v0.3" ]]; then
    # ARCH=Mistral
    FEEDBACK="[INST] Your answer is incorrect.[/INST]"
    SKIP_PATH=${SCRIPT_DIR}/../../metadata/mistral_${DATASET}_correct_cases.json
    BASE_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
else
    echo "${RED}Do not supprt ${MODEL}${NC}"
    exit
fi

if [[ "${DATASET}" == "MBPP" ]]; then
    VERSION=${DATASET}_${MODEL}_seed42_trials10
    EPOCH=10
elif [[ "${DATASET}" == "GSM8K" ]]; then
    VERSION=${DATASET}_${MODEL}_seed42_trials10
    EPOCH=3
elif [[ "${DATASET}" == "MATH500" ]]; then
    VERSION=${DATASET}_${MODEL}_seed42_trials10
    EPOCH=3
else
    echo "${RED}Do not supprt ${DATASET}${NC}"
    exit
fi

# Create directory
mkdir -p ${OUTPUT_DIR}
cp "$0" ${OUTPUT_DIR}/

# Run training
# torchrun --standalone --nproc-per-node=${NGPU} --nnodes=1 --master-port $((RANDOM % (65535 - 10000 + 1) + 10000)) ${SCRIPT_DIR}/../run.py train \
python ${SCRIPT_DIR}/../run.py train \
    --output_dir ${OUTPUT_DIR} \
    --dataset ${DATASET} \
    --data_version ${VERSION} \
    --fp16 \
    --correct_trial_strategy "include" \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --seed ${SEED} \
    --feedback_str "${FEEDBACK}" \
    --feedback_type gt \
    --inner_params ${INNER_PARAMS} \
    --norm \
    --rank 16 \
    --mlp_class LoRA \
    --act dropout:0.1 \
    --shared \
    --batch_size ${LOCAL_BATCH_SIZE} \
    --num_train_epochs ${EPOCH} \
    --gradient_accumulation $(( ${GLOBAL_BATCH_SIZE} / ${LOCAL_BATCH_SIZE} / ${NGPU} )) \
    --logging_steps 10 \
    --combine | tee -a ${OUTPUT_DIR}/train.log

# Run evaluation
python ${SCRIPT_DIR}/../run.py eval \
    --output_dir ${OUTPUT_DIR}/epoch-${EPOCH} \
    --dataset ${DATASET} \
    --model ${BASE_MODEL_PATH} \
    --seed ${SEED} \
    --feedback_type gt \
    --max_new_tokens 2048 \
    --skip_cases_path ${SKIP_PATH} \
    --batch_size ${LOCAL_BATCH_SIZE} 2>&1 | tee -a ${OUTPUT_DIR}/epoch-${EPOCH}/eval.log
