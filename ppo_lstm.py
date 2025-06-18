import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim

from CryptoTradingEnv_cleaned_essential import CryptoTradingEnv

from ppo_essential_notes import PPOBuffer, device


def reset_env(env):
    """Reset environment returning only observation."""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    """Step environment supporting gym or gymnasium API."""
    result = env.step(action)
    if len(result) == 5:
        next_obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_obs, reward, done, info = result
    return next_obs, reward, done, info


class LSTMGaussianActor(nn.Module):
    """Simple Gaussian policy with a single LSTM layer."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x: torch.Tensor, hx=None):
        # Accept either (B, obs_dim) or (B, T, obs_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, obs_dim)
        out, hx = self.lstm(x, hx)
        mu = self.mu(out.squeeze(1))  # (B, act_dim)
        std = torch.exp(self.log_std)
        return Normal(mu, std), hx


class LSTMValue(nn.Module):
    """State-value function with an LSTM backbone."""

    def __init__(self, obs_dim: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, hx=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, obs_dim)
        out, hx = self.lstm(x, hx)
        value = self.v(out.squeeze(1))  # (B, 1)
        return value.squeeze(-1), hx


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.actor = LSTMGaussianActor(obs_dim, act_dim, hidden_size)
        self.critic = LSTMValue(obs_dim, hidden_size)

    def act(self, obs, hx_a=None, hx_c=None):
        dist, hx_a = self.actor(obs, hx_a)
        value, hx_c = self.critic(obs, hx_c)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, hx_a, hx_c

    def get_logp_value(self, obs, action, hx_a=None, hx_c=None):
        dist, hx_a = self.actor(obs, hx_a)
        logp = dist.log_prob(action).sum(-1)
        value, hx_c = self.critic(obs, hx_c)
        return logp, value, hx_a, hx_c




def ppo_update(ac, data, pi_opt, v_opt, clip_ratio=0.2):
    logp_old = data['logp']
    adv = (data['adv'] - data['adv'].mean()) / (data['adv'].std() + 1e-8)

    for _ in range(10):
        dist, _ = ac.actor(data['obs'])
        logp = dist.log_prob(data['act']).sum(-1)
        ratio = (logp - logp_old).exp()
        clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clipped)).mean()

        pi_opt.zero_grad()
        loss_pi.backward()
        pi_opt.step()

    for _ in range(10):
        value, _ = ac.critic(data['obs'])
        loss_v = ((value - data['ret']) ** 2).mean()
        v_opt.zero_grad()
        loss_v.backward()
        v_opt.step()


def train(env=None, env_name='Pendulum-v1', env_kwargs=None, steps_per_epoch=2048, epochs=50, hidden_size=64):
    """Train an LSTM-based PPO agent."""
    if env is None:
        if env_name == 'CryptoTradingEnv':
            env = CryptoTradingEnv(**(env_kwargs or {}))
        else:
            env = gym.make(env_name)
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    act_dim = env.action_space.shape[0]
    ac = ActorCritic(obs_dim, act_dim, hidden_size).to(device)

    pi_opt = optim.Adam(ac.actor.parameters(), lr=3e-4)
    v_opt = optim.Adam(ac.critic.parameters(), lr=1e-3)
    buffer = PPOBuffer(obs_dim, act_dim, steps_per_epoch)

    for epoch in range(epochs):
        obs = reset_env(env)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        hx_a = hx_c = None
        for t in range(steps_per_epoch):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, value, hx_a, hx_c = ac.act(obs_tensor, hx_a, hx_c)
            act = action.cpu().numpy()[0]
            next_obs, rew, done, _ = step_env(env, act)
            buffer.store(obs, act, rew, value.cpu().numpy(), logp.cpu().numpy())
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            if done or (t == steps_per_epoch - 1):
                if done:
                    last_val = 0
                else:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        last_val, _ = ac.critic(obs_tensor, hx_c)
                        last_val = last_val.cpu().numpy()
                buffer.finish_path(last_val)
                if done:
                    obs = reset_env(env)
                    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                    hx_a = hx_c = None
        data = buffer.get()
        ppo_update(ac, data, pi_opt, v_opt)
        print(f'Epoch {epoch+1} completed')


if __name__ == '__main__':
    try:
        env = CryptoTradingEnv()
        train(env=env, env_name='CryptoTradingEnv')
    except Exception as e:
        print('Failed to initialize CryptoTradingEnv:', e)
        print('Falling back to Pendulum-v1')
        train()
