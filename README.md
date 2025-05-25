# Parameter-Efficient Fine-Tuning of RoBERTa for News Classification Using LoRA

This repository presents our solution for the Deep Learning Project 2 (Spring 2025), where we fine-tuned a RoBERTa model using Low-Rank Adaptation (LoRA) on the AG News dataset, under a strict constraint of **fewer than 1 million trainable parameters**.

## Project Overview

The objective was to apply LoRA to specific attention layers of a pre-trained RoBERTa model for efficient text classification. The constraint enforced that the total trainable parameters not exceed 1 million. We achieved this using HuggingFace's PEFT (Parameter-Efficient Fine-Tuning) library and extensive experimentation on hyperparameters and LoRA configuration.

**Final Results:**
- Validation Accuracy: 92.71%
- Public Test Accuracy (Kaggle): 84.73%
- Private Test Accuracy (Kaggle): 83.75%
- Total Trainable Parameters: 999,172

## Model Architecture

- **Base Model**: `roberta-base` (12 transformer layers, 768 hidden dimensions)
- **LoRA Rank**: 4
- **Scaling Factor (α)**: 16
- **Target Modules**: Attention query, value, and output projection layers
- **LoRA Dropout**: 0.05
- **Sequence Length**: 128 tokens
- **Tokenizer**: RoBERTa tokenizer with `add_prefix_space=True`

This configuration was selected to provide a balance between expressiveness and efficiency while remaining within the parameter budget.

## Methodology

### Preprocessing
- Normalization of newline and HTML characters
- Tokenization and padding to a fixed length of 128
- Uniform input dimensions across batches

### Training Strategy
- **Epochs**: 3
- **Batch Size**: 8 (training), 64 (evaluation)
- **Optimizer**: AdamW (β1=0.9, β2=0.999, ε=1e-8)
- **Scheduler**: Cosine with restarts
- **Warmup Ratio**: 0.15
- **Loss Function**: CrossEntropy with Label Smoothing (0.05)
- **Gradient Accumulation**: 4 steps
- **Precision**: Mixed precision (bfloat16)
- **Early Stopping**: Enabled based on validation loss

## Key Findings

- LoRA provides scalable, high-performance fine-tuning for downstream tasks with <1% of the original model's parameters.
- Targeting only key layers (query, value, output) is sufficient for competitive results.
- Hyperparameters such as learning rate schedules and gradient accumulation had significant impact on performance.
- Regularization (dropout, label smoothing) helped control overfitting.

## Limitations and Future Work

While our current model performed well, potential areas of improvement include:
- Exploring dynamic sequence lengths or hierarchical encoding
- Layer-wise tuning of LoRA rank and scaling factor
- Extended training beyond 3 epochs with adaptive scheduling
- Lightweight data augmentation and class balancing
- Ensembling and test-time augmentation strategies

## Repository Contents

- `finalcode.ipynb`: Jupyter notebook containing data preprocessing, model implementation, training pipeline, and evaluation
- Full training logs, loss curves, and accuracy trends
- Inference scripts and prediction outputs (optional)

## Team Members

- Neha Patil 
- Ruchi Jha 
- Satvik Upadhyay 

## License

This repository is intended for academic purposes as part of NYU's Deep Learning course. All implementation and experimentation were conducted by the authors unless otherwise cited.
