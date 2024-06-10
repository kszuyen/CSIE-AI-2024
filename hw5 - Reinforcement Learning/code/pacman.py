import os
import time
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl_algorithm import DQN
from custom_env import ImageEnv
from utils import seed_everything, YOUR_CODE_HERE
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # environment hyperparameters
    parser.add_argument("--env_name", type=str, default="ALE/MsPacman-v5")
    parser.add_argument("--state_dim", type=tuple, default=(4, 84, 84))
    parser.add_argument("--image_hw", type=int, default=84, help="The height and width of the image")
    parser.add_argument("--num_envs", type=int, default=4)
    # DQN hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    parser.add_argument("--target_update_interval", type=int, default=10000)
    # training hyperparameters
    parser.add_argument("--max_steps", type=int, default=int(2.5e5))
    parser.add_argument("--eval_interval", type=int, default=1000)
    # others
    parser.add_argument("--save_root", type=Path, default="./submissions")
    parser.add_argument(
        "--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment"
    )
    # evaluation
    parser.add_argument("--eval", action="store_true", help="evaluate the model")
    parser.add_argument("--eval_model_path", type=str, default=None, help="the path of the model to evaluate")
    return parser.parse_args()


def validation(agent, num_evals=5):
    eval_env = gym.make("ALE/MsPacman-v5")
    eval_env = ImageEnv(eval_env)

    scores = 0
    for i in range(num_evals):
        (state, _), done = eval_env.reset(), False
        while not done:
            # do action from your agent
            # action = YOUR_CODE_HERE
            action = agent.act(state, training=False)  # Use epsilon-greedy without exploration

            # get your action feedback from environment
            # next_state, reward, terminated, truncated, info = YOUR_CODE_HERE
            next_state, reward, terminated, truncated, info = eval_env.step(action)

            state = next_state
            scores += reward
            done = terminated or truncated
    return np.round(scores / num_evals, 4)


def train(agent, env):
    history = {"Step": [], "AvgScore": [], "ValueLoss": []}
    maxScore = 0
    (state, _) = env.reset()

    for _ in tqdm(range(args.max_steps)):

        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        result = agent.process(
            (state, action, reward, next_state, terminated)
        )  # You can track q-losses over training from `result` variable.

        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

        if agent.total_steps % args.eval_interval == 0:
            avg_score = validation(agent)
            history["Step"].append(agent.total_steps)
            history["AvgScore"].append(avg_score)
            history["ValueLoss"].append(result["value_loss"])

            # log info to plot your figure
            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax1.plot(history["Step"], history["AvgScore"], "b-", label="Average Score")
            ax2.plot(history["Step"], history["ValueLoss"], "r-", label="Value Loss")
            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Average Score", color="b")
            ax2.set_ylabel("Value Loss", color="r")
            ax1.set_title("Training Progress")
            ax1.grid(True)
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(save_dir / "training_plot.png")
            plt.close()
            # save model
            if avg_score > maxScore:
                maxScore = avg_score
                # torch.save(agent.network.state_dict(), save_dir / "pacman_dqn_best.pt")
                torch.save(agent.network.state_dict(), save_dir / "pacman_dqn.pt")
            print("Step: {}, AvgScore: {}, ValueLoss: {}".format(agent.total_steps, avg_score, result["value_loss"]))


def evaluate(agent, eval_env, capture_frames=True):
    seed_everything(0, eval_env)  # don't modify

    # load the model
    if agent is None:
        action_dim = eval_env.action_space.n
        state_dim = (args.num_envs, args.image_hw, args.image_hw)
        agent = DQN(state_dim=state_dim, action_dim=action_dim)
        agent.network.load_state_dict(torch.load(args.eval_model_path))

    (state, _), done = eval_env.reset(), False

    scores = 0
    # Record the frames
    if capture_frames:
        writer = imageio.get_writer(save_dir / "mspacman.mp4", fps=10)

    while not done:
        if capture_frames:
            writer.append_data(eval_env.render())
        else:
            eval_env.render()

        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, info = eval_env.step(action)
        state = next_state
        scores += reward
        done = terminated or truncated
    if capture_frames:
        writer.close()
    print("The score of the agent: ", scores)


def main():
    env = gym.make(args.env_name)
    env = ImageEnv(env, stack_frames=args.num_envs, image_hw=args.image_hw)

    action_dim = env.action_space.n
    state_dim = (args.num_envs, args.image_hw, args.image_hw)
    agent = DQN(state_dim=state_dim, action_dim=action_dim)

    # train
    train(agent, env)

    # evaluate
    eval_env = gym.make(args.env_name, render_mode="rgb_array")
    eval_env = ImageEnv(eval_env, stack_frames=args.num_envs, image_hw=args.image_hw)
    evaluate(agent, eval_env)


if __name__ == "__main__":
    args = parse_args()

    # save_dir = args.save_root / f"{args.env_name.replace('/', '-')}__{args.exp_name}__{int(time.time())}"
    save_dir = args.save_root
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    if args.eval:
        eval_env = gym.make(args.env_name, render_mode="rgb_array")
        eval_env = ImageEnv(eval_env, stack_frames=args.num_envs, image_hw=args.image_hw)
        evaluate(agent=None, eval_env=eval_env, capture_frames=True)
    else:
        main()
