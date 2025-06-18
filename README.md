# Reinforcement-Learning
Implementation of Reinforcement Learning For Financial Assets Allocation

## Running the LSTM PPO example

Install the required packages (e.g. `gym`, `torch`, and `pandas`) and run

```bash
python ppo_lstm.py
```

This attempts to train on `CryptoTradingEnv` if it can be instantiated,
falling back to the standard `Pendulum-v1` environment otherwise.
