# Guarda como run_experiments_save_rewards.py
# Uso: python run_experiments_save_rewards.py --episodes 300 --runs 5 --out results_with_rewards.csv

import argparse, csv, random
import gymnasium as gym
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from collections import deque, namedtuple
Experience = namedtuple("Experience", ("s","a","r","s2","done"))

class ReplayBuffer:
    def __init__(self, capacity): self.buf=deque(maxlen=capacity)
    def push(self, *args): self.buf.append(Experience(*args))
    def sample(self, n): return random.sample(self.buf, n)
    def __len__(self): return len(self.buf)

class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers=[]
        in_dim=input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim,h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class Agent:
    def __init__(self, obs_dim, n_actions, hidden_sizes, device, lr=1e-3, gamma=0.99, batch=64):
        self.device=device
        self.online = DQNNet(obs_dim, n_actions, hidden_sizes).to(device)
        self.target = DQNNet(obs_dim, n_actions, hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.crit = nn.MSELoss()
        self.gamma = gamma
        self.batch = batch
        self.replay = ReplayBuffer(5000)
        self.n_actions = n_actions
    def act(self, s, eps):
        if random.random() < eps: return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            return int(self.online(t).argmax(1).item())
    def update(self):
        if len(self.replay) < self.batch: return None
        batch = self.replay.sample(self.batch)
        s = torch.FloatTensor([b.s for b in batch]).to(self.device)
        a = torch.LongTensor([b.a for b in batch]).unsqueeze(1).to(self.device)
        r = torch.FloatTensor([b.r for b in batch]).to(self.device)
        s2 = torch.FloatTensor([b.s2 for b in batch]).to(self.device)
        done = torch.FloatTensor([0.0 if b.done else 1.0 for b in batch]).to(self.device)
        q = self.online(s).gather(1,a).squeeze(1)
        with torch.no_grad():
            qnext = self.target(s2).max(1)[0]
            target = r + self.gamma * qnext * done
        loss = self.crit(q, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())
    def sync(self): self.target.load_state_dict(self.online.state_dict())

def run_one(cfg, episodes=300, target_update=50, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    agent = Agent(obs_dim, env.action_space.n, cfg["hidden_sizes"], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), lr=cfg["lr"], gamma=cfg["gamma"], batch=cfg["batch"])
    eps = cfg.get("eps_start",1.0); eps_end = cfg.get("eps_end",0.01); eps_decay = cfg.get("eps_decay",0.995)
    rewards = []
    losses = []
    for ep in range(episodes):
        state,_ = env.reset(seed=seed)
        ep_r = 0.0
        for t in range(500):
            a = agent.act(state, eps)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.replay.push(state, a, r, ns, done)
            l = agent.update()
            if l is not None: losses.append(l)
            state = ns; ep_r += r
            if done: break
        rewards.append(ep_r)
        eps = max(eps_end, eps * eps_decay)
        if (ep+1) % target_update == 0:
            agent.sync()
    env.close()
    return {"rewards": rewards, "mean_last20": float(np.mean(rewards[-20:])), "mean_all": float(np.mean(rewards)), "max": float(np.max(rewards)), "loss_mean": float(np.mean(losses)) if losses else None}

def main():
    import argparse, json
    parser=argparse.ArgumentParser()
    parser.add_argument("--episodes",type=int,default=300)
    parser.add_argument("--runs",type=int,default=3)
    parser.add_argument("--out",type=str,default="results_with_rewards.csv")
    args=parser.parse_args()
    experiments = [
        {"name":"3x128","hidden_sizes":[128,128,128],"lr":1e-4,"gamma":0.99,"batch":64},
        {"name":"2x128","hidden_sizes":[128,128],"lr":1e-3,"gamma":0.99,"batch":64},
    ]
    rows=[]
    for cfg in experiments:
        for run in range(args.runs):
            out = run_one(cfg, episodes=args.episodes, target_update=50, seed=100+run)
            rows.append({
                "exp": cfg["name"],
                "run": run+1,
                "mean_last20": out["mean_last20"],
                "mean_all": out["mean_all"],
                "max": out["max"],
                "std": np.std(out["rewards"]),
                "loss_mean": out["loss_mean"],
                "rewards": json.dumps(out["rewards"])
            })
            print(f"Exp {cfg['name']} run {run+1} mean_last20={out['mean_last20']:.2f} max={out['max']:.1f}")
    # guardar CSV
    keys = list(rows[0].keys())
    import csv
    with open(args.out,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Guardado en", args.out)

if __name__=="__main__":
    main()
