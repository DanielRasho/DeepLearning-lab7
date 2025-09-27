#!/usr/bin/env python3
"""
Script minimal para correr varios experimentos DQN en CartPole,
cambiando la arquitectura de la red (número de capas y hidden_size).
Genera un CSV con resultados resumidos por experimento.
"""

import argparse
import csv
import random
from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

Experience = namedtuple("Experience", ("s", "a", "r", "s2", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Experience(*args))
    def sample(self, n):
        return random.sample(self.buf, n)
    def __len__(self):
        return len(self.buf)

class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, hidden_sizes, device, lr=1e-3, gamma=0.99, batch_size=64, buffer_capacity=5000):
        self.device = device
        self.online = DQNNet(obs_dim, n_actions, hidden_sizes).to(device)
        self.target = DQNNet(obs_dim, n_actions, hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_capacity)
        self.n_actions = n_actions

    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online(s)
            return int(q.argmax(1).item())

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        s = torch.FloatTensor([b.s for b in batch]).to(self.device)
        a = torch.LongTensor([b.a for b in batch]).to(self.device).unsqueeze(1)
        r = torch.FloatTensor([b.r for b in batch]).to(self.device)
        s2 = torch.FloatTensor([b.s2 for b in batch]).to(self.device)
        done = torch.FloatTensor([0.0 if b.done else 1.0 for b in batch]).to(self.device)

        q_vals = self.online(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            target = r + self.gamma * q_next * done
        loss = self.criterion(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

def run_one(config, episodes=200, target_update=50, max_steps=500, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(obs_dim, n_actions, config["hidden_sizes"], device,
                     lr=config["lr"], gamma=config["gamma"], batch_size=config["batch"])
    eps = config.get("eps_start", 1.0)
    eps_end = config.get("eps_end", 0.01)
    eps_decay = config.get("eps_decay", 0.995)
    rewards = []
    losses = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed)
        ep_reward = 0.0
        for t in range(max_steps):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay.push(state, action, reward, next_state, done)
            l = agent.update()
            if l is not None:
                losses.append(l)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        eps = max(eps_end, eps * eps_decay)
        if (ep + 1) % target_update == 0:
            agent.sync_target()
    env.close()
    # summary stats
    return {
        "mean_reward": float(np.mean(rewards[-20:])),  # media últimos 20
        "mean_reward_all": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "std_reward": float(np.std(rewards)),
        "loss_mean": float(np.mean(losses)) if losses else None,
        "rewards": rewards
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3, help="repeats per config")
    parser.add_argument("--out", type=str, default="results.csv")
    args = parser.parse_args()

    # Grid de arquitecturas / hiperparámetros a probar
    experiments = [
        {"name": "2x128", "hidden_sizes": [128, 128], "lr": 1e-3, "gamma": 0.99, "batch": 64},
        {"name": "2x64",  "hidden_sizes": [64, 64],  "lr": 1e-3, "gamma": 0.99, "batch": 64},
        {"name": "3x128", "hidden_sizes": [128, 128, 128], "lr": 1e-4, "gamma": 0.99, "batch": 64},
        {"name": "1x256", "hidden_sizes": [256], "lr": 1e-3, "gamma": 0.99, "batch": 64},
    ]

    rows = []
    for cfg in experiments:
        print("Running experiment:", cfg["name"])
        for run in range(args.runs):
            out = run_one(cfg, episodes=args.episodes, target_update=50, seed=42+run)
            print(f"  run {run+1}/{args.runs} -> mean(last20)={out['mean_reward']:.2f} max={out['max_reward']:.1f}")
            rows.append({
                "exp": cfg["name"],
                "run": run+1,
                "mean_last20": out["mean_reward"],
                "mean_all": out["mean_reward_all"],
                "max": out["max_reward"],
                "std": out["std_reward"],
                "loss_mean": out["loss_mean"]
            })
    # guardar CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Resultados guardados en", args.out)

if __name__ == "__main__":
    main()
