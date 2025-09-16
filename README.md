# Fine-tuning Mistral 7B using QLoRA

A comprehensive tutorial for fine-tuning the Mistral 7B model using Quantized Low-Rank Adaptation (QLoRA) on the ViGGO dataset for meaning representation tasks. 

### We use the [Nebius AI Cloud](https://nebius.com/services/studio-inference-service?utm_medium=cpc&utm_source=yoloco&utm_campaign=harpercarrollai).

## Overview

This project demonstrates how to fine-tune the powerful [Mistral 7B](https://github.com/mistralai/mistral-src) model using QLoRA, a memory-efficient fine-tuning technique that combines quantization and LoRA (Low-Rank Adaptation). The tutorial uses the GEM/viggo dataset to teach the model to generate structured meaning representations from natural language sentences.

## Key Features

- **Memory Efficient**: Uses 4-bit quantization with bitsandbytes
- **Parameter Efficient**: QLoRA fine-tuning with only 0.56% trainable parameters
- **Practical Example**: Trains on meaning representation tasks that showcase clear learning
- **Cost Effective**: Complete training for under $3 using cloud GPU

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: H100 with 200GB memory)
- Jupyter Notebook environment

## Installation

Install the required packages:

```bash
apt-get update && apt-get install -y -q git
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q -U datasets==2.14.6 scipy ipywidgets
pip install -q -U pyarrow==20.0.0
```

## Dataset

This tutorial uses the [GEM/viggo](https://huggingface.co/datasets/GEM/viggo) dataset, which contains:
- **Training samples**: 5,103
- **Validation samples**: 714  
- **Test samples**: 1,083

The dataset maps natural language sentences to structured meaning representations, making it ideal for demonstrating fine-tuning effectiveness.

## Model Architecture

### Base Model
- **Model**: `mistralai/Mistral-7B-v0.1`
- **Quantization**: 4-bit with BitsAndBytesConfig
- **Parameters**: 3.77B total parameters

### LoRA Configuration
- **Rank (r)**: 8
- **Alpha**: 16
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head`
- **Dropout**: 0.05
- **Trainable parameters**: 21.26M (0.56% of total)

## Training Configuration

```python
TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=1000,
    learning_rate=2.5e-5,
    warmup_steps=5,
    logging_steps=50,
    save_steps=50,
    eval_steps=50,
    bf16=True,
    optim="paged_adamw_8bit"
)
```

## Usage

1. **Setup Environment**: Follow the cloud setup instructions for Nebius AI Cloud or similar GPU provider

2. **Load and Prepare Data**:
```python
from datasets import load_dataset
train_dataset = load_dataset('GEM/viggo', split='train')
eval_dataset = load_dataset('GEM/viggo', split='validation')
```

3. **Configure Model**:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

4. **Run Training**: Execute the training loop with the configured parameters

5. **Evaluate**: Test the fine-tuned model against the base model performance

## Example Input/Output

**Input Sentence**: 
> "Earlier, you stated that you didn't have strong feelings about PlayStation's Little Big Adventure. Is your opinion true for all games which don't have multiplayer?"

**Expected Output**: 
> `verify_attribute(name[Little Big Adventure], rating[average], has_multiplayer[no], platforms[PlayStation])`

**Base Model Output**: Verbose, incorrect JSON structure

**Fine-tuned Model Output**: Correct structured meaning representation

## Results

The fine-tuned model successfully learns to:
- Generate structured meaning representations
- Use correct function names from the specified vocabulary
- Include appropriate attributes and values
- Follow the exact format expected by the dataset

Training loss decreased from 0.75 to 0.15 over 1000 steps, with validation loss stabilizing around 0.154.

## Cloud Setup (Nebius AI)

The tutorial includes detailed instructions for setting up a training environment on Nebius AI Cloud:

1. **Docker Configuration**:
```bash
# Docker run configuration
-p 8888:8888 --restart=always --gpus all --shm-size=16GB

# Docker image  
pytorch/pytorch:latest

# Entrypoint
bash -c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

2. **Access**: Connect via `http://[YOUR-VM-IP]:8888`

## Important Notes

⚠️ **Cost Management**: Remember to stop or delete your cloud instances when done to avoid unnecessary charges.

⚠️ **Memory Management**: The notebook includes instructions for restarting kernels between training and inference to manage GPU memory effectively.

## File Structure

```
├── mistral-finetune-nebius.ipynb    # Main tutorial notebook
├── mistral-viggo-finetune/          # Output directory with checkpoints
│   ├── checkpoint-50/
│   ├── checkpoint-100/
│   ├── ...
│   └── checkpoint-1000/             # Final model
└── logs/                            # Training logs
```

## Contributing

This tutorial is designed to be educational and reproducible. Feel free to:
- Experiment with different hyperparameters
- Try different datasets
- Extend the training for longer convergence
- Compare with other fine-tuning methods

## License

This project follows the licensing terms of the constituent libraries (Transformers, PEFT, BitsAndBytes) and the Mistral model license.

## Acknowledgments

- Tutorial created by [Harper Carroll](https://github.com/harper-carroll)
- Based on the Mistral 7B model by Mistral AI
- Uses the ViGGO dataset from the GEM benchmark
- Utilizes QLoRA methodology for efficient fine-tuning

## Social Links

- [Instagram](https://instagram.com/harpercarrollai)
- [X/Twitter](https://x.com/harperscarroll)
- [YouTube](https://youtube.com/@harpercarrollai)
- [LinkedIn](https://linkedin.com/in/harpercarroll)
- [TikTok](https://tiktok/@harpercarrollai)
