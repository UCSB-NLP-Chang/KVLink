# KVLink: Accelerating LLMs via Efficient KV Cache Reuse  

This is the official implementation of the paper **"KVLink: Accelerating LLMs via Efficient KV Cache Reuse."**

## **Table of Contents**  
1. [Preparation](#preparation)  
   - [Virtual Environment Setup](#virtual-environment-setup)
   - [Data Preprocessing](#data-preprocessing) 
2. [Implementation](#implementation)  
3. [Model Training](#model-training)  
   - [Configuration Management](#configuration-management)  
   - [Download the Tokenizer and Model](#download-the-tokenizer-and-model)  
     - [Download the Tokenizer](#download-the-tokenizer)  
     - [Download the Model](#download-the-model)  
   - [Run the Training](#run-the-training)  
   - [Logging and Visualization](#logging-and-visualization)  
     - [View TensorBoard Logs on a Remote Server](#view-tensorboard-logs-on-a-remote-server)  
4. [Evaluation](#evaluation)  
   - [Convert Model Checkpoint to PyTorch Format](#convert-model-checkpoint-to-pytorch-format)  
   - [Run Evaluation](#run-evaluation)  
   - [Run NQ Evaluation](#run-nq-evaluation)  
5. [Acknowledgement](#acknowledgement)

---

## Preparation

<a id="virtual-environment-setup"></a>
### 1. Virtual Environment Setup

The training code is built upon [torchtitan](https://github.com/pytorch/torchtitan) and [torchtune](https://github.com/pytorch/torchtune), which provide optimized implementations to improve training efficiency.

To set up the environment, follow these steps:

1. **Install the preview version of PyTorch** (required for `torchtitan`):

   ```bash
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

2. **Install additional dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install `torchtune` and this repository**:

   ```bash
   pip install torchtune
   pip install -e ./
   ```



<a id="data-preprocessing"></a>
### 2. Data Preprocessing

The data preprocessing scripts are provided under the [`scripts/data_process/`](scripts/data_process/) directory. Each script contains documentation and usage instructions in its header.

#### 2.1 Pre-training Data Preparation
To preprocess the pre-training data, run:

```bash
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
```

#### 2.2 SFT/Multi-turn Conversation/Summarization Data

```bash
python scripts/data_process/daring_anteater.py --max_length=4096 --validation_size=2000
python scripts/data_process/tulu.py --max_length=4096 --validation_size=2000
python scripts/data_process/sum.py --max_length=4096 --validation_size=1000
```

#### 2.3 QA Data Preparation

Our QA training data is built upon the **2WikiMultiHopQA** and **TriviaQA** datasets. To access the original **2WikiMultiHopQA** dataset, use:

```bash
git clone https://huggingface.co/datasets/xanhho/2WikiMultihopQA
```

For the **TriviaQA** dataset, run:

```bash
git clone https://github.com/facebookresearch/FiD
cd FiD
bash get-data.sh 
```

After obtaining the datasets, retrieve relevant QA documents using **Contriever** and generate answers using **GPT-4** by running:

```babashsh
python scripts/data_process/gpt_answer.py
```

Lastly, run the preprocessing script as other datasets

```bash
python scripts/data_process/block_qa.py --max_length=4096 --validation_size=2000
```

---

## Implementation

The **cross-document reconnection with summary tokens** is implemented in [`src/data/titan_preprocess.py`](src/data/titan_preprocess.py), specifically at **Line 21**, where the preprocessor adds summary tokens to each context segment.

The **special attention mask** (illustrated in **Figure 2** of the paper) is implemented in the function [`make_segment_mask`](src/data/titan_preprocess.py#L591), located at **Line 591** of [`src/data/titan_preprocess.py`](src/data/titan_preprocess.py).

---

## Model Training

The training process is primarily based on **torchtitan** (see [`titan_trainer_kvlink.py`](titan_trainer_kvlink.py)).

<a id="configuration-management"></a>
### 0. Configuration Management
Instead of using external configuration files in `YAML` or `JSON` format, or setting hyperparameters via command-line arguments, we define all training configurations directly in the code. Our rationale is that hyperparameters coud be the same important as implementation as the model architecture and algorithms. Also, defining them in code improves readability and maintainability by keeping all experiment settings in one place without relying on external config files or command-line arguments.


To simplify experiment management, the training script takes a single command-line parameter:  

```bash
--config_name <config_name>
```

This `config_name` maps to a predefined set of hyperparameters for a specific experiment. For example, the configuration **`datav6_step6k_bsz64_reencode_5_selective_ckpt`** corresponds to the following setup:

- **Data mixture version:** `v6` (defined in [`src/training/titan_training_utils.py`](src/training/titan_training_utils.py), under `DATASET_MAPPING`)
- **Training steps:** `6000`
- **Batch size:** `64`
- **Link tokens:** `5`
- **Selective gradient checkpointing:** Enabled (to save memory)
- **Learning rate:** `5e-6`

This approach ensures that each experiment remains reproducible and well-documented within the codebase.

<a id="download-the-tokenizer-and-model"></a>
### **1. Download the Tokenizer and Model**

#### **Download the Tokenizer**
Run the following command to download the tokenizer:

```bash
python src/data/titan_download_tokenizer.py \
    --repo_id meta-llama/Llama-3.2-1B-Instruct \
    --tokenizer_path "original" \
    --local_dir data/titan_tokenizer/ \
    --hf_token=YOUR_HF_TOKEN
```

To use `Llama-3.2-3B-Instruct`, replace `repo_id` with `meta-llama/Llama-3.2-3B-Instruct`

#### **Download the Model**
Run the following command to download the model:

```bash
tune download meta-llama/Llama-3.2-1B-Instruct \
    --output-dir model_cache/Llama-3.2-1B-Instruct \
    --ignore-patterns "original/consolidated.00.pth" \
    --hf-token YOUR_HF_TOKEN
```

Similarly, for the `3B` model, replace `meta-llama/Llama-3.2-1B-Instruct` with `meta-llama/Llama-3.2-3B-Instruct`

---

<a id="run-the-training"></a>
### **2. Run the Training**
Run the following script to start training:

```bash
LOG_RANK=${LOG_RANK:-0}
NGPU=${NGPU:-"8"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    titan_trainer.py --config_name datav6_step6k_bsz64_reencode_5_selective_ckpt
```

---

<a id="logging-and-visualization"></a>
### **3. Logging and Visualization**
By default, training logs are saved using **TensorBoard** in `{job_dump_folder}` (as defined in the training config). 

To enable **Weights & Biases (`wandb`)**, add the flag `--use_wandb_for_log` to the training command.

#### **View TensorBoard Logs on a Remote Server**
Run the following command on your local machine:

```bash
ssh -L 6006:localhost:6006 your_remote@remote_address
```

Then, start **TensorBoard** on the remote server:

```bash
tensorboard --logdir=/path/to/KVMemory/run_logs/block_datav1_step10k/tensorboard/20250105-1953 --port=6006
```

Now, open **[http://localhost:6006/](http://localhost:6006/)** in your local browser.

---

## Evaluation

The evaluation code is available in [`scripts/evaluation`](scripts/evaluation).

### **Convert Model Checkpoint to PyTorch Format**
Since `torchtitan` saves models in **DCP format**, you need to convert the saved checkpoint to **PyTorch format** before evaluation. Use the following command:

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch \
    /path/to/checkpoint/step-1000 /save_path/to/pytorch_checkpoint.pt
```

- `/path/to/checkpoint/step-1000` – Directory containing the saved `DCP` checkpoints.  
- `/save_path/to/pytorch_checkpoint.pt` – Destination path for the converted PyTorch model.

### **Run Evaluation**
Once the checkpoint is converted, run the Wikipedia evaluation script:

```bash
python scripts/evaluation/hqa_eval.py \
    --ckpt_path checkpoint.pt \
    --batch_size 10 \
    --reencode_num 5 \
    --attn_type "blocked"
```

### **Run NQ Evaluation**
For **Natural Questions (NQ) evaluation**, an additional `--pos` argument (ranging from 0 to 9) is required to specify the golden document index:

```bash
python scripts/evaluation/nq_eval.py \
    --ckpt_path checkpoint.pt \
    --batch_size 10 \
    --pos 0 \
    --reencode_num 5 \
    --attn_type "blocked"
```

If a HuggingFace pretrained model is used, the argument 'hf' should be added. For example,

```
python scripts/evaluation/hqa_eval.py \
    --ckpt_path "meta-llama/Llama-3.2-1B-Instruct" \
    --batch_size 10 \
    --reencode_num 5 \
    --attn_type "blocked" \
    --hf True
```

The KVLink5 models in the paper are available at [KVLink5-1B](https://huggingface.co/Shiyu-Lab/Llama1B-KVLink5) and [KVLink5-3B](https://huggingface.co/Shiyu-Lab/Llama3B-KVLink5).

## Acknowledgement

Our training script is mainly based on torchtitan and torchtune:

```
@misc{torchtitan,
      title={TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training},
      author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
      year={2024},
      eprint={2410.06511},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.06511},
}
```

```
@software{torchtune,
  title = {torchtune: PyTorch's finetuning library},
  author = {torchtune maintainers and contributors},
  url = {https//github.com/pytorch/torchtune},
  license = {BSD-3-Clause},
  month = apr,
  year = {2024}
}
```
