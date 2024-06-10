# Homework 5: Reinforcement Learning

## [Assignment Guidelines](https://docs.google.com/presentation/d/1enBQeAZwnNpqga9D7C8EezGplVjcS8T4/edit?usp=sharing&ouid=113209338386021225150&rtpof=true&sd=trueLinks)

## Install Necessary Packages
```
conda create -n hw5 python=3.11 -y
conda activate hw5
pip install -r requirements.txt
```

## Run Training Process
```
python pacman.py
```

- This runs the training process with default hyperparameters. To reproduce the results, simply run with the default hyperparameters.
- The training results will be saved in the `/submissions`folder as default, including the training plot and the best model's weight.
  
## Run Evaluation
```
python pacman.py --eval --eval_model_path submissions/pacman_dqn.pt
```