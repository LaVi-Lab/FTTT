export RM_API=http://175.102.19.159:6003
export OJ_API=http://localhost:9999

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET=$2
MODEL=$1
SEED=${3:-42}
TRIALS=1
REPEAT=10
TRAIN_SIZE=12000

# if [[ "${DATASET}" == "MBPP" || "${DATASET}" == "HumanEval" ]]; then
#     FEEDBACK_TYPE=gt
# else
#     FEEDBACK_TYPE=rm
# fi
FEEDBACK_TYPE=gt


if [[ "${MODEL}" == "Llama-3.1-8B-Instruct" ]]; then
    REFLECT_STR="<|start_header_id|>user<|end_header_id|>\n\nYour answer is incorrect. Please carefully check the solution and summarize all mistakes in short. Do NOT provide the corrected solution. Do NOT say \"my solution\".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    BASE_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
elif [[ "${MODEL}" == "Mistral-7B-Instruct-v0.3" ]]; then
    REFLECT_STR="[INST] Your answer is incorrect. Carefully check the solution step-by-step and list all mistakes in short. MUST NOT provide the correct answer. Your response MUST be in the third person tone. [/INST]"
    BASE_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
else
    echo "${RED}Do not supprt ${MODEL}${NC}"
    exit
fi

NUM_GEN=$(( ${TRIALS} * ${REPEAT} ))

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
python ${SCRIPT_DIR}/../run.py gen \
    --dataset ${DATASET} \
    --feedback_type ${FEEDBACK_TYPE} \
    --data_version ${DATASET}_${MODEL}_seed${SEED}_trials${NUM_GEN}_size${TRAIN_SIZE} \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --seed ${SEED} \
    --num_return_sequences ${TRIALS} \
    --repeat_generate ${REPEAT} \
    --reflect_str "${REFLECT_STR}" \
    --max_train_size ${TRAIN_SIZE} \
    --batch_size 1
