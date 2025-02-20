# Learning to Reason from Feedback at Test-Time

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is the official repository of the paper "*Learning to Reason from Feedback at Test-Time*".

## 🛠️ Installation

Our implementation is based on `python=3.12`. Follow the commands below to prepare the Python environment (we recommend using [Miniconda](https://docs.anaconda.com/miniconda/) to setup the environment):

```bash
git clone https://github.com/LaVi-Lab/FTTT.git
cd FTTT
conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 💡 Preparation

### ⏬ Data

- We mainly train and evaluate on MATH500, GSM8K, MBPP and HumanEval. You can download the raw data via the links below or choose to download our packed version on [Google Drive](https://drive.google.com/file/d/1MvaNSC1HsuegfvAvOwqppPr7bntp-3LN/view?usp=sharing):

    | Dataset | Link |
    | :------ | :--: |
    | MATH500 | [<img src="https://skillicons.dev/icons?i=github" alt="GitHub Icon" width="20" height="20">](https://github.com/openai/prm800k/tree/main/prm800k/math_splits) |
    | GSM8K | [<img src="https://skillicons.dev/icons?i=github" alt="GitHub Icon" width="20" height="20">](https://github.com/openai/grade-school-math) |
    | MBPP | [🤗](https://huggingface.co/datasets/google-research-datasets/mbpp) |
    | HumanEval | [🤗](https://huggingface.co/datasets/openai/openai_humaneval) |

    > [!NOTE]
    > All downloaded datasets should be stored in a folder named `datasets`.

- Another important asset of this repo is the question indices of the test sets, as we only evaluate hard questions that cannot be solved by the raw LLM initially. You can download these indices from [Google Drive](https://drive.google.com/file/d/1YrPXcXG_-4OTgo7HlujpTfSQU1XU0Gev/view?usp=sharing) and unzip them to the folder `metadata`.

    <details>
    <summary>Reproducing question indices</summary>

    If you want to reproduce the question indices by yourself, you can perform greedy decoding on each dataset:

    ```bash
    # Greedy decoding
    bash prelim/scripts/greedy.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval]
    ```

    This command will automatically save a file `{llama|mistral}_{MATH500|GSM8K|MBPP|HumanEval}_correct_cases.json` under the current directory, which contains the indices of questions that can be correctly answered by greedy decoding.

    </details>

- The training data of OpTune consists of solutions generated from the raw LLM. We provide the [Google Drive](https://drive.google.com/file/d/1wCSInPlNrziS1JF66GpEX-AFRjWo4Tvd/view?usp=sharing) link to download our training data.

    <details>
    <summary>Reproducing OpTune training data</summary>

    You can generate the training data for OpTune by yourself:

    ```bash
    bash optim/scripts/gen.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP]
    ```

    </details>

Finally, all data should be organized as follows:

```
FTTT
|-- datasets [Raw training & evaluation data here!]
|-- metadata [Test set indices here!]
|-- api
|-- optim
|   |-- cache [OpTune Training data here!]
|   |-- data
|   |-- models
|   |-- scripts
|   |-- ...
|-- prelim
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
```

### 🌐 API

For the code generation task, we deploy a local API service to run the generated code and check if it passes all test cases. We use the following commands to launch the service:

```bash
cd api
uvicorn oj_api:app --host 0.0.0.0 --port 9999 --workers 8 --limit-concurrency 16
```

By default, this codebase will send requests to `http://localhost:9999` for evaluating code completion datasets. If you want to use another port or host, please modify `--host` and `--port` above and add `export OJ_API=YOUR_NEW_URL` to our scripts to make the new URL effective.

## 📊 FTTT Experiments

Below are commands to reproduce all of our experiments on FTTT and other test-time scaling baselines:

```bash
# Revision
bash prelim/scripts/revision.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval]
# Beam Search
bash prelim/scripts/beam_search.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval]
# Beam Search Sampling
bash prelim/scripts/beam_search_sampling.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval]
# Best-of-N
bash prelim/scripts/best_of_n.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval] [42|85|100]
# Self-Refine
bash prelim/scripts/self_refine.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval] [42|85|100]
# Self-Consistency
bash prelim/scripts/self_consistency.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval] [42|85|100]
# FTTT (w/o or w/ self-reflection)
bash prelim/scripts/fttt.sh [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [MATH500|GSM8K|MBPP|HumanEval] [42|85|100] [FTTT|FTTT+]
```

> [!NOTE]
> All output logs will be stored in the folder `outputs` under the current directory.

## 🎯 OpTune Experiments

### 📌 Training

You can use the following commands to reproduce the training of PEFT baselines as well as OpTune:

```bash
# PEFT baselines
bash optim/scripts/baseline.sh [MATH500|GSM8K|MBPP] 42 [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3] [FT|LoRA|Adapter|IA3|LNTuning]
# OpTune
bash optim/scripts/train.sh [MATH500|GSM8K|MBPP] 42 [Llama-3.1-8B-Instruct|Mistral-7B-Instruct-v0.3]
```

> [!NOTE]
> All checkpoints will be stored in the folder `outputs` under the current directory.

### ⚖️ Testing

After training the models, you can use the following commands to evaluate the trained PEFT baselines and OpTune:

```bash
# PEFT baselines
bash optim/scripts/baseline_scale.sh ${PEFT_CHECKPOINT_PATH}
# OpTune
bash optim/scripts/scale.sh ${OPTUNE_CHECKPOINT_PATH}
```

Typically, we use the checkpoints in the last epoch for evaluation.

To reproduce our OpTune performance, you can download our checkpoints for each dataset and run the evaluation:

<center>

| Dataset | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.3 |
| :------ | :--: | :--: |
| MATH500 | [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/512px-Google_Drive_icon_%282020%29.svg.png" alt="Google Drive Icon" width="20" height="20">](https://drive.google.com/file/d/1M5AniWkXtD17PvrPls3Q3Ma6WUfKFQtn/view?usp=sharing) | [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/512px-Google_Drive_icon_%282020%29.svg.png" alt="Google Drive Icon" width="20" height="20">](https://drive.google.com/file/d/1DtFijlLIash6Z1aBN1yy3McPgD5SjF76/view?usp=sharing) |
| GSM8K | [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/512px-Google_Drive_icon_%282020%29.svg.png" alt="Google Drive Icon" width="20" height="20">](https://drive.google.com/file/d/1RCmD00_9_BDPIexL2W9FcZwSu-HxQhmp/view?usp=sharing) | [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/512px-Google_Drive_icon_%282020%29.svg.png" alt="Google Drive Icon" width="20" height="20">](https://drive.google.com/file/d/1dD7TsMTf7DYWp0GzHgYZpYkWey80-UGU/view?usp=sharing) |
| MBPP | [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/512px-Google_Drive_icon_%282020%29.svg.png" alt="Google Drive Icon" width="20" height="20">](https://drive.google.com/file/d/15oomWNg3sEoTDKl0FBaZBjCva7Jookkz/view?usp=sharing) | [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/512px-Google_Drive_icon_%282020%29.svg.png" alt="Google Drive Icon" width="20" height="20">](https://drive.google.com/file/d/12Furc43EewaOPlJBssgRb7SGeGX11fVB/view?usp=sharing) |

</center>

## 🤝 Acknowledgement

Some of our implementations are based on [MEND](https://github.com/eric-mitchell/mend). We would like to thank the authors for sharing their code.