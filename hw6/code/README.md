# 2024 AI hw6 - Reinforce Learning with Human Feedback

### Assignment Link: [Link](https://docs.google.com/presentation/d/1AqHzkpCJDnuq90Lt9VnfqGenoSApb3RfPeXJGS-PkO0/edit?usp=sharing)
- Due to environmental issues, I use **"Google Colab Pro"** for this assignment.
- GPU: **L4 GPU**
- To reproduce the results, follow **AI_hw6.ipynb** or simply follow the steps below in Google Colab.

## 1. Install Packages
```
!pip3 install --no-deps --upgrade transformers==4.41.1 datasets==2.19.1 accelerate==0.30.1 bitsandbytes==0.43.1 trl==0.8.6 peft==0.11.1
!pip3 install tqdm packaging wandb
```

### Check if GPU is available, and the version of PyTorch and Cuda
```
import torch
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
```

## 2. Install Unsloth
```
%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers
```

### Check if all installation is successfull
```
!nvcc -V
!python -m xformers.info
!python -m bitsandbytes
```

## 3. Mount to Google drive, and change directory to the uploaded folder.
```
from google.colab import drive
drive.mount('/content/drive/')
%cd /content/drive/MyDrive/AI2024-hw6
```

## 4. Generate text from the selected LLM: unsloth/gemma-2b-bnb-4bit
```
!bash inference.sh unsloth/gemma-2b-bnb-4bit <wandb_token>
```

## 5. Finetune with DPO
```
!bash run.sh DPO unsloth/gemma-2b-bnb-4bit <wandb_token> 0.1
```

## 6. Fintune with ORPO
```
!bash run.sh ORPO unsloth/gemma-2b-bnb-4bit <wandb_token>
```

## 7. Extra Experiments #1

### Fintune DPO, ORPO with different hyperparameter: beta=0.2
```
!bash run.sh DPO unsloth/gemma-2b-bnb-4bit <wandb_token> 0.2 beta02
!bash run.sh ORPO unsloth/gemma-2b-bnb-4bit <wandb_token> 0.2 beta02
````

## 8. Extra Experiments #2

### Select and compare with different model: "unsloth/tinyllama-bnb-4bit"
```
!bash inference.sh unsloth/tinyllama-bnb-4bit <wandb_token>
!bash run.sh DPO unsloth/tinyllama-bnb-4bit <wandb_token> 0.1
!bash run.sh ORPO unsloth/tinyllama-bnb-4bit <wandb_token> 0.1
```
