set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_VERBOSITY=error

export OJ_API=http://localhost:9999

MODEL=$1
DATASET=$2

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

python prelim/revision.py \
    --model_name_or_path ${MODEL_DIR}  \
    --model_arch ${ARCH} \
    --dataset ${DATASET} \
    --temperature 0 \
    --num_trials 1 2>&1 | tee outputs/${MODEL}_${DATASET}_greedy.log
