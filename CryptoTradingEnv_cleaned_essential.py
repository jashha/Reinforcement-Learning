import random
import numpy as np
import torch
import gym
import pandas as pd

def set_seed(seed, env=None):
    """
    Set random seed for reproducibility across random, numpy, torch, and (optionally) a Gym environment.

    Args:
        seed (int): Random seed to set.
        env: Gym environment (optional).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform

class HierarchicalRiskParity:
    def __init__(self, returns):
        self.returns = returns
        self.cov_matrix = returns.cov()
        self.corr_matrix = returns.corr()
        
    def _compute_distance_matrix(self):
        # Convert correlation matrix to distance matrix
        return np.sqrt(0.5 * (1 - self.corr_matrix))
    
    def _compute_hierarchical_tree(self):
        # Compute the linkage matrix
        distance_matrix = self._compute_distance_matrix()
        linkage_matrix = linkage(squareform(distance_matrix), method='ward') # 'single')
        return linkage_matrix
    
    def _get_quasi_diagonal_cluster_sequence(self, linkage_matrix):
        # Get the order of rows/columns after rearranging them to put similar items together
        return leaves_list(linkage_matrix)
    
    def _recursive_bisection(self, cov_matrix, order):
        # Base case: If only one asset, return its weight as 1
        if len(order) == 1:
            return [1]
        
        # Split the assets into two clusters
        split = len(order) // 2
        cluster1, cluster2 = order[:split], order[split:]
        
        # Compute the inverse variances
        cluster1_inv_variance = 1 / np.sum(np.diag(cov_matrix.iloc[cluster1, cluster1]))
        cluster2_inv_variance = 1 / np.sum(np.diag(cov_matrix.iloc[cluster2, cluster2]))
        
        # Allocate capital between the two clusters based on inverse variance
        alpha = 1 - cluster1_inv_variance / (cluster1_inv_variance + cluster2_inv_variance)
        
        # Recursively assign weights within each cluster
        return list(np.array(self._recursive_bisection(cov_matrix, cluster1)) * alpha) + \
               list(np.array(self._recursive_bisection(cov_matrix, cluster2)) * (1-alpha))
    
    def get_hrp_weights(self):
        # Compute hierarchical tree
        linkage_matrix = self._compute_hierarchical_tree()
        
        # Get the quasi-diagonal cluster sequence
        order = self._get_quasi_diagonal_cluster_sequence(linkage_matrix)
        
        # Apply recursive bisection to compute portfolio weights
        hrp_weights = self._recursive_bisection(self.cov_matrix, order)
        # print(f'||| order: {order} ||| hrp_weights: {hrp_weights} |||')
        # stop
        
        # Create a pandas Series for the weights, indexed by asset names
        # return pd.Series(hrp_weights, index=self.returns.columns[order])
        return hrp_weights, order, pd.Series(hrp_weights, index=self.returns.columns[order])





import math

# --- try to import SciPy specials, otherwise provide fallbacks ---
try:
    from scipy.special import gammaln, digamma
except ImportError:
    # fallback for log-gamma
    from math import lgamma as gammaln

    # fallback for digamma (ψ) using recurrence + asymptotic expansion
    def digamma(x: float) -> float:
        """
        Approximate digamma(x). For x < 6, use recurrence ψ(x) = ψ(x+1) - 1/x
        to shift into region x >= 6, then use asymptotic series.
        """
        result = 0.0
        # recurrence to bump x up to >= 6
        while x < 6:
            result -= 1.0 / x
            x += 1.0
        # asymptotic expansion for large x
        inv = 1.0 / x
        inv2 = inv * inv
        # include terms up to O(1/x^6)
        result += (
            math.log(x)
            - 0.5 * inv
            - inv2 * (1.0/12 - inv2 * (1.0/120 - inv2 * (1.0/252)))
        )
        return result

# --- the BlockProbabilityTracker class ---
class BlockProbabilityTracker:
    """
    Tracks a Dirichlet posterior over per-asset block-probabilities.
    Conjugate update: Dirichlet(α) + Multinomial(k) → Dirichlet(α + k).
    """

    def __init__(self, n_assets: int, prior: float = 1.0):
        """
        Args:
          n_assets: number of assets
          prior:    symmetric Dirichlet prior α₀ for each asset
        """
        self.n_assets = n_assets
        self.prior    = prior
        self.reset()

    def reset(self):
        """Reset α back to the symmetric prior (call at episode start)."""
        self.alpha = np.full(self.n_assets, self.prior, dtype=float)

    def track(self, blocked_counts: np.ndarray):
        """
        Incorporate new block observations (Multinomial counts).
        blocked_counts[i] = # times asset i was blocked this step.
        """
        assert blocked_counts.shape == (self.n_assets,)
        self.alpha += blocked_counts

    def posterior_mean(self) -> np.ndarray:
        """
        Posterior mean of the block probabilities:
          p_i = α_i / (Σ_j α_j)
        """
        total = self.alpha.sum()
        return self.alpha / total

    def posterior_variance(self) -> np.ndarray:
        """
        Marginal variance of each p_i under the Dirichlet posterior:
          Var[p_i] = α_i (α₀ − α_i) / [α₀² (α₀ + 1)],  where α₀ = Σ_j α_j.
        """
        a0    = self.alpha.sum()
        denom = a0**2 * (a0 + 1)
        if denom > 1e-8:
            return (self.alpha * (a0 - self.alpha)) / denom
        else:
            return np.zeros(self.n_assets, dtype=float)

    def posterior_entropy(self) -> float:
        """
        Differential entropy of the Dirichlet posterior:

          H[Dir(α)] 
            = ln B(α) + (α₀ − K) ψ(α₀) 
              − Σ_i (α_i − 1) ψ(α_i)

        where
          α₀ = Σ_i α_i,
          B(α) = ∏_i Γ(α_i) / Γ(α₀),
          ψ(·)  = digamma.
        """
        a0 = self.alpha.sum()
        K  = self.n_assets

        # log B(α) = Σ_i ln Γ(α_i) − ln Γ(α₀)
        log_B = np.sum(gammaln(self.alpha)) - gammaln(a0)

        # H = log B + (α₀ − K)·ψ(α₀) − Σ (α_i − 1)·ψ(α_i)
        ent = (
            log_B
            + (a0 - K) * digamma(a0)
            - np.dot(self.alpha - 1, digamma(self.alpha))
        )
        return float(ent)

    def __repr__(self):
        mean = np.round(self.posterior_mean(), 3).tolist()
        var  = np.round(self.posterior_variance(), 5).tolist()
        ent  = round(self.posterior_entropy(), 3)
        return (
            f"<BlockTracker α={np.round(self.alpha,3).tolist()}, "
            f"mean={mean}, var={var}, ent={ent}>"
        )

    def __str__(self):
        mean = self.posterior_mean()
        mean_str = ", ".join(f"{m:.2f}" for m in mean)
        return f"Block probs ≈ [{mean_str}]"





        


def prefill_replay_buffer_and_scalers(
    agent, env, test_env, replay_buffer, replay_buffer_path,
    scaler_path, running_scaler_path, replay_buffer_size, device,
    max_ep_len, logger=None, is_ppo=False,
):
    """
    Pre-fill the replay buffer with random experience, and fit scalers.
    Will skip if files already exist.
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print('obs_dim:', obs_dim, 'act_dim', act_dim)
    scaler_path = f'obs_dim_{obs_dim}_' + f'act_dim_{act_dim}' + scaler_path
    running_scaler_path = f'obs_dim_{obs_dim}_' + f'act_dim_{act_dim}' + running_scaler_path
    replay_buffer_path = f'obs_dim_{obs_dim}_' + f'act_dim_{act_dim}' + replay_buffer_path
    print(f'||| scaler_path: {scaler_path} ||| running_scaler_path: {running_scaler_path} ||| replay_buffer_path: {replay_buffer_path} |||')
    # stop
    # Check if files already exist
    buffer_exists = os.path.exists(replay_buffer_path)
    scaler_exists = os.path.exists(scaler_path)
    running_scaler_exists = os.path.exists(running_scaler_path)

    # Check if files is valid
    buffer_exists = is_valid_pickle(replay_buffer_path)
    scaler_exists = is_valid_pickle(scaler_path)
    running_scaler_exists = is_valid_pickle(running_scaler_path)

    # Remove corrupt files if needed
    if not buffer_exists and os.path.exists(replay_buffer_path):
        os.remove(replay_buffer_path)
    if not scaler_exists and os.path.exists(scaler_path):
        os.remove(scaler_path)
    if not running_scaler_exists and os.path.exists(running_scaler_path):
        os.remove(running_scaler_path)


    # Load if already exists
    # if buffer_exists and scaler_exists and running_scaler_exists:
    # if buffer_exists and scaler_exists and running_scaler_exists and is_ppo:
    if scaler_exists and running_scaler_exists and is_ppo:
        # try:
        #     replay_buffer.load(replay_buffer_path)
        # except (EOFError, pickle.UnpicklingError):
        #     logger.warning("Replay buffer file corrupt or empty, refilling.")
        #     os.remove(replay_buffer_path)
        try:
            scaler = load_pickle(scaler_path)
            env.set_scaler(scaler)
            test_env.set_scaler(scaler)
        except (EOFError, pickle.UnpicklingError):
            logging.warning("Scaler file corrupt or empty, refilling.")
            os.remove(scaler_path)
        try:
            running_scaler = load_pickle(running_scaler_path)
            env.set_running_scaler(running_scaler)
            test_env.set_running_scaler(running_scaler)
        except (EOFError, pickle.UnpicklingError):
            logging.warning("Running Scaler file corrupt or empty, refilling.")
            os.remove(running_scaler_path)

        # replay_buffer.load(replay_buffer_path) # Original.
        # scaler = load_pickle(scaler_path)
        # running_scaler = load_pickle(running_scaler_path)
        # env.set_scaler(scaler)
        # env.set_running_scaler(running_scaler)
        # test_env.set_scaler(scaler)
        # test_env.set_running_scaler(running_scaler)
        if logging:
            logging.info("Scalers loaded from disk!!!!!!!!!!!")
        return running_scaler, scaler

    elif scaler_exists and running_scaler_exists and is_ppo:
    # elif buffer_exists and scaler_exists and running_scaler_exists and not is_ppo:
        try:
            scaler = load_pickle(scaler_path)
            env.set_scaler(scaler)
            test_env.set_scaler(scaler)
        except (EOFError, pickle.UnpicklingError):
            logging.warning("Scaler file corrupt or empty, refilling.")
            os.remove(scaler_path)
        try:
            running_scaler = load_pickle(running_scaler_path)
            env.set_running_scaler(running_scaler)
            test_env.set_running_scaler(running_scaler)
        except (EOFError, pickle.UnpicklingError):
            logging.warning("Running Scaler file corrupt or empty, refilling.")
            os.remove(running_scaler_path)
        if logger:
            logging.info("Scalers loaded from disk!!!!!!!!!!!")
        return running_scaler, scaler # , _

    # Otherwise, pre-fill
    elif is_ppo:
    # else:
        # range_ = int(replay_buffer_size // max_ep_len) # Original.
        # range_ = int(replay_buffer_size // max_ep_len // 500) # 5000) # 2000) # 200)
        range_ = int(replay_buffer_size // max_ep_len) # * 30 # 1 # 2
        total_steps = int(max_ep_len * range_)
        progress_bar = tqdm(total=total_steps, desc="PPO Pre-Training Progress")

        if logging:
            logging.info(f'Prefilling replay buffer: range_={range_}, max_ep_len={max_ep_len}')

        all_hist_trades = []  # Collect features across episodes
        for t in range(range_):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            # while not (ep_len == max_ep_len-1):
            while not (d or (ep_len == max_ep_len)):
                # if not is_ppo:
                a = env.action_space.sample()
                # else:
                #     a = agent.ac.act(torch.as_tensor(o).to(device, non_blocking=True))
                o2, r, d, _ = env.step(a)
                o = o2
                ep_len += 1
                progress_bar.update(1)
                if d and not (ep_len == max_ep_len):
                    o, d, ep_ret = env.reset(), False, 0
                if ep_len == max_ep_len - 1:
                    break
                print(f'||| ep_len: {ep_len} ||| max_ep_len: {max_ep_len} ||| actions: {a}')
                print()

            # Fit scalers every episode (optional)
            historical_trades = env.historical_trades.replace([-np.inf, np.inf], np.nan).fillna(value=0)
            # After episode, collect features from env.historical_trades
            ep_trades = env.historical_trades.copy() # .replace([-np.inf, np.inf], np.nan).dropna()

            all_hist_trades.append(ep_trades)
            dataframe = env.data.set_index('timestamp', drop=True, inplace=False).replace([-np.inf, np.inf], np.nan).dropna()
            if hasattr(env, 'get_running_scaler'):
                # running_scaler = env.get_running_scaler()
                env.running_scaler.fit(historical_trades)
            if hasattr(env, 'get_scaler'):
                # scaler = env.get_scaler()
                env.scaler.fit(dataframe)

        progress_bar.close()

        # Concatenate all feature DataFrames
        all_trades_df = pd.concat(all_hist_trades, axis=0, ignore_index=True)

        # Drop inf/nan, ensure all feature columns are present
        all_trades_df = all_trades_df.replace([np.inf, -np.inf], np.nan).dropna()
        print("Fitting scaler on shape:", all_trades_df.shape)

        running_scaler = env.get_running_scaler()
        # Fit running scaler (e.g., StandardScaler) on all aggregated data
        running_scaler.fit(all_trades_df) # .values)
        # Save/copy scaler to your env/test_env as needed

        # Save scalers
        if hasattr(env, 'get_scaler'):
            scaler = env.get_scaler()
            save_pickle(scaler, scaler_path)
            train_data = env.data
            scaler.fit(train_data.set_index('timestamp'))
            test_env.set_scaler(scaler)
            env.set_scaler(scaler)
        if hasattr(env, 'get_running_scaler'):
            # running_scaler = env.get_running_scaler()
            # Fit running scaler (e.g., StandardScaler) on all aggregated data
            running_scaler.fit(all_trades_df) # .values)
            # Save/copy scaler to your env/test_env as needed
            save_pickle(running_scaler, running_scaler_path)
            test_env.set_running_scaler(running_scaler)
            env.set_running_scaler(running_scaler)
        ep_trades.to_csv('sac_env_bak_historical_trades_sample.csv')
        save_pickle(scaler, scaler_path)
        save_pickle(running_scaler, running_scaler_path)
        if logging:
            logging.info("Scalers prefilled and saved.")

        return running_scaler, scaler

    # Otherwise, pre-fill
    if not is_ppo:
        scaler_path = f'sac_obs_dim_{obs_dim}_' + f'act_dim_{act_dim}_' + scaler_path
        running_scaler_path = f'sac_obs_dim_{obs_dim}_' + f'act_dim_{act_dim}_' + running_scaler_path
        replay_buffer_path = f'sac_obs_dim_{obs_dim}_' + f'act_dim_{act_dim}_' + replay_buffer_path
        print(f'||| scaler_path: {scaler_path} ||| running_scaler_path: {running_scaler_path} ||| replay_buffer_path: {replay_buffer_path} |||')

        # Check if files already exist
        buffer_exists = os.path.exists(replay_buffer_path)
        scaler_exists = os.path.exists(scaler_path)
        running_scaler_exists = os.path.exists(running_scaler_path)

        # Check if files is valid
        buffer_exists = is_valid_pickle(replay_buffer_path)
        scaler_exists = is_valid_pickle(scaler_path)
        running_scaler_exists = is_valid_pickle(running_scaler_path)

        # Remove corrupt files if needed
        if not buffer_exists and os.path.exists(replay_buffer_path):
            os.remove(replay_buffer_path)
        if not scaler_exists and os.path.exists(scaler_path):
            os.remove(scaler_path)
        if not running_scaler_exists and os.path.exists(running_scaler_path):
            os.remove(running_scaler_path)


        # Load if already exists
        # if buffer_exists and scaler_exists and running_scaler_exists:
        if scaler_exists and running_scaler_exists:
            # try:
            #     replay_buffer.load(replay_buffer_path)
            # except (EOFError, pickle.UnpicklingError):
            #     logger.warning("Replay buffer file corrupt or empty, refilling.")
            #     os.remove(replay_buffer_path)
            try:
                scaler = load_pickle(scaler_path)
                env.set_scaler(scaler)
                test_env.set_scaler(scaler)
            except (EOFError, pickle.UnpicklingError):
                logger.warning("Scaler file corrupt or empty, refilling.")
                os.remove(scaler_path)
            try:
                running_scaler = load_pickle(running_scaler_path)
                env.set_running_scaler(running_scaler)
                test_env.set_running_scaler(running_scaler)
            except (EOFError, pickle.UnpicklingError):
                logger.warning("Running Scaler file corrupt or empty, refilling.")
                os.remove(running_scaler_path)

            # replay_buffer.load(replay_buffer_path) # Original.
            # scaler = load_pickle(scaler_path)
            # running_scaler = load_pickle(running_scaler_path)
            # env.set_scaler(scaler)
            # env.set_running_scaler(running_scaler)
            # test_env.set_scaler(scaler)
            # test_env.set_running_scaler(running_scaler)
            if logger:
                logger.info("Scalers loaded from disk!!!!!!!!!!!")
            return running_scaler, scaler # , replay_buffer


        # range_ = int(replay_buffer_size // max_ep_len) # Original. 
        range_ = int(replay_buffer_size // max_ep_len) #  // 1000) # 00 # 00) # 5000) # 2000) # 200)
        total_steps = int(max_ep_len * 1) # range_) # 1)
        progress_bar = tqdm(total=total_steps, desc="SAC Pre-Training Progress")

        if logger:
            logger.info(f'Prefilling replay buffer: range_={range_}, max_ep_len={max_ep_len}')

        all_hist_trades = []  # Collect features across episodes

        for t in range(1): # range_):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            # while not (d or (ep_len == max_ep_len)): # Original.
            while not (ep_len == max_ep_len):
            # if not (ep_len == max_ep_len):
                a = env.action_space.sample()
                o2, r, d, _ = env.step(a)
                replay_buffer.store(obs=o, act=a, rew=r, next_obs=o2, done=d)
                o = o2
                ep_len += 1
                progress_bar.update(1)
                if d and not (ep_len == max_ep_len):
                    o, d, ep_ret = env.reset(), False, 0
                if ep_len == max_ep_len - 1:
                    break
                print(f'||| ep_len: {ep_len} ||| max_ep_len: {max_ep_len} ||| actions: {a}')
                print()
                # if d:
                #     if ep_len != max_ep_len:
                #         o, d, ep_ret, ep_len = env.reset(), False, 0, 0

            # Fit scalers every episode (optional)
            # historical_trades = env.historical_trades.replace([-np.inf, np.inf], np.nan).dropna()
            historical_trades = env.historical_trades.replace([-np.inf, np.inf], np.nan).fillna(value=0)
            # After episode, collect features from env.historical_trades
            ep_trades = env.historical_trades.copy() # .replace([-np.inf, np.inf], np.nan).dropna()

            all_hist_trades.append(ep_trades)
            dataframe = env.data.set_index('timestamp', drop=True, inplace=False).replace([-np.inf, np.inf], np.nan).dropna()
            if hasattr(env, 'get_running_scaler'):
                running_scaler = env.get_running_scaler()
                running_scaler.fit(historical_trades)
            if hasattr(env, 'get_scaler'):
                scaler = env.get_scaler()
                scaler.fit(dataframe)

        progress_bar.close()

        # Concatenate all feature DataFrames
        all_trades_df = pd.concat(all_hist_trades, axis=0, ignore_index=True)

        # Drop inf/nan, ensure all feature columns are present
        all_trades_df = all_trades_df.replace([np.inf, -np.inf], np.nan).dropna()
        print("Fitting scaler on shape:", all_trades_df.shape)

        # Fit running scaler (e.g., StandardScaler) on all aggregated data
        running_scaler.fit(all_trades_df) # .values)
        # Save/copy scaler to your env/test_env as needed

        # Save buffer and scalers
        # replay_buffer.save(replay_buffer_path) # Original.

        if hasattr(env, 'get_scaler'):
            scaler = env.get_scaler()
            save_pickle(scaler, scaler_path)
            train_data = env.data
            scaler.fit(train_data.set_index('timestamp'))
            test_env.set_scaler(scaler)
            env.set_scaler(scaler)
        if hasattr(env, 'get_running_scaler'):
            # running_scaler = env.get_running_scaler()
            # Fit running scaler (e.g., StandardScaler) on all aggregated data
            running_scaler.fit(all_trades_df) # .values)
            # Save/copy scaler to your env/test_env as needed
            save_pickle(running_scaler, running_scaler_path)
            test_env.set_running_scaler(running_scaler)
            env.set_running_scaler(running_scaler)
        ep_trades.to_csv('sac_env_bak_historical_trades_sample.csv')
        save_pickle(scaler, scaler_path)
        save_pickle(running_scaler, running_scaler_path)
        if logger:
            logger.info("Replay buffer and scalers prefilled and saved.")

        return running_scaler, scaler
        # return running_scaler, scaler, replay_buffer




from typing import List, Optional, Dict
from itertools import product








class FeatureEngine:
    """
    Computes rolling risk and alpha features on a dynamic portfolio.
    """
    def __init__(
        self,
        # window=252,
        window_size=30,
        maxlen=10000,
        compute_market_beta=True,
        asset='Close',
        prefix="pf",
        min_trade_value=10.0, # Minimum notional trade value to execute
        risk_free_rate=0.000118, #   -> Daily risk free rate. # (0.0463) -> yearly risk free rate.
        ema_short_period: int = 12,
        ema_long_period: int = 26,
        rsi_window: int = 14,
        entropy_bins: int = 10,
        var_ratio_lag: int = 5,
        horizon: Optional[int] = None,
    ):
        ratio = int(window_size // 30)
        self.asset = asset
        self.horizon = horizon
        # Buffers for new metrics
        self.realized_pnls: List[float] = []
        self.cumulative_returns: List[float] = []

        # Buffers for new features
        self.cash_ratios: List[float] = []
        # self.asset_counts: List[float] = []
        self.concentration_ratios: List[float] = []
        self.trade_counts: List[float] = []
        self.value_zscores: List[float] = []

        self.entropy_bins = entropy_bins #  * ratio 
        self.var_ratio_lag = var_ratio_lag * ratio if ratio != 0. else int(window_size // 2)
        self.momentum_window = window_size # momentum_window or seq_len
        
        self.min_trade_value = min_trade_value  # 10.0      # Minimum notional trade value to execute
        # self.engine.reset()
        self.risk_free_rate_ = risk_free_rate
        self.asset_value = float(0.)
        self.prev_asset_value = float(0.)
        self.returns = []

        # EMA parameters
        self.ema_short_period = ema_short_period * ratio if ratio != 0. else int(window_size // 2)
        self.ema_long_period = ema_long_period * ratio if ratio != 0. else int(window_size // 2)
        self.alpha_short = 2 / (ema_short_period * ratio + 1) if ratio != 0. else int(window_size // 2)
        self.alpha_long = 2 / (ema_long_period * ratio + 1) if ratio != 0. else int(window_size // 2)
        self.prev_ema_short: Optional[float] = None
        self.prev_ema_long: Optional[float] = None

        # RSI window
        self.rsi_window = rsi_window

        self.mean = None
        self.std = None
        self._peak_value = None
        self._peak_index = None
        self._max_drawdown = 0.0
        self._max_drawdown_length = 0
        self._current_drawdown_start = None
        self._current_drawdown_start: Optional[int] = None

        self.weights_history: List[np.ndarray] = []
        self.benchmark_returns: List[float] = []
        self.drawdowns: List[float] = []
        self.vol_history: List[float] = []
        self.asset_values = []
        self.should_be_asset_values = []

        self.cash_ratios : List[float] = []
        # self.asset_counts : List[float] = []
        self.concentration_ratios : List[float] = []
        self.trade_counts : List[float] = []
        self.value_zscores : List[float] = []
        self.realized_pnls : List[float] = []
        self.cumulative_returns : List[float] = []
        self.price_history : List[float] = []
        self.momentum_history : List[float] = []
        
        self.should_be_asset_value = float(0.)
        self.should_be_asset_values = []

        self.window = window_size
        self.maxlen = maxlen
        self.prefix = prefix
        self.asset = asset
        self.compute_market_beta = compute_market_beta

        self.asset_values = [] # deque(maxlen=maxlen)
        self.market_returns = deque(maxlen=maxlen)
        self.risk_free_rates = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
        self.latest_features = {}
        self.metrics = {}
        self.static_risk_free = risk_free_rate

        self.portfolio_excess_returns = [] # deque(maxlen=maxlen) # []
        self.benchmark_excess_returns = [] # deque(maxlen=maxlen) # []

        """self.col_names = [
            f"{self.asset}_{self.window}_alpha_monthly",
            f"{self.asset}_{self.window}_vol_monthly",
            f"{self.asset}_{self.window}_skew",
            f"{self.asset}_{self.window}_kurtosis",
            f"{self.asset}_{self.window}_cvar_05",
            f"{self.asset}_{self.window}_sharpe",
            f"{self.asset}_{self.window}_sortino",
            f"{self.asset}_{self.window}_max_drawdown",
            f"{self.asset}_{self.window}_m_squared_ratio",
            f"{self.asset}_{self.window}_beta",
            f"{self.asset}_{self.window}_beta_adj_sharpe",
            ]"""



        self.col_names = [
            f"{self.asset}_{self.window}_sharpe",
            f"{self.asset}_{self.window}_sortino",
            f"{self.asset}_{self.window}_momentum",
            f"{self.asset}_{self.window}_skewness",
            f"{self.asset}_{self.window}_kurtosis",
            f"{self.asset}_{self.window}_realized_vol",
            f"{self.asset}_{self.window}_ulcer_index",
            f"{self.asset}_{self.window}_vol_of_vol",
            f"{self.asset}_{self.window}_beta",
            f"{self.asset}_{self.window}_correlation",
            f"{self.asset}_{self.window}_omega",
            f"{self.asset}_{self.window}_calmar_mean_r",
            f"{self.asset}_{self.window}_var",
            f"{self.asset}_{self.window}_cvar",
            f"{self.asset}_{self.window}_max_drawdown",
            f"{self.asset}_{self.window}_max_drawdown_length",
            # f"{self.asset}_{self.window}_turnover",
            # f"{self.asset}_{self.window}_transaction_cost",
            # f"{self.asset}_{self.window}_entropy",
            # f"{self.asset}_{self.window}_risk_parity_deviation",
            f"{self.asset}_{self.window}_excess_return",
            f"{self.asset}_{self.window}_tracking_error",
            f"{self.asset}_{self.window}_information_ratio",
            f"{self.asset}_{self.window}_ema_short",
            f"{self.asset}_{self.window}_ema_long",
            # f"{self.asset}_{self.window}_macd",
            f"{self.asset}_{self.window}_rsi",
            # f"{self.asset}_{self.window}_win_rate",
            f"{self.asset}_{self.window}_profit_factor",
            f"{self.asset}_{self.window}_rachev",
            f"{self.asset}_{self.window}_calmar_cum_r",
            f"{self.asset}_{self.window}_drawdown_recovery_time",
            f"{self.asset}_{self.window}_autocorr_lag1",
            f"{self.asset}_{self.window}_hurst",
            f"{self.asset}_{self.window}_efficiency_ratio",
            f"{self.asset}_{self.window}_kelly_fraction",
            f"{self.asset}_{self.window}_last_return_z",
            f"{self.asset}_{self.window}_price_zscore",
            f"{self.asset}_{self.window}_mfe",
            f"{self.asset}_{self.window}_mae",
            # f"{self.asset}_{self.window}_tail_variance",
            # f"{self.asset}_{self.window}_downside_vol",
            f"{self.asset}_{self.window}_momentum_zscore",
            f"{self.asset}_{self.window}_return_entropy",
            f"{self.asset}_{self.window}_variance_ratio",
            # f"{self.asset}_{self.window}_win_streak",
            # f"{self.asset}_{self.window}_loss_streak",
            # f"{self.asset}_{self.window}_effective_bets",
            f"{self.asset}_{self.window}_mad",
            f"{self.asset}_{self.window}_gini",
            # f"{self.asset}_{self.window}_hill_estimator",
            f"{self.asset}_{self.window}_drawdown_at_risk",
            f"{self.asset}_{self.window}_cov",
            # f"{self.asset}_{self.window}_cash_ratio",
            # f"{self.asset}_{self.window}_asset_count",
            # f"{self.asset}_{self.window}_concentration_ratio",
            # f"{self.asset}_{self.window}_trade_count",
            f"{self.asset}_{self.window}_asset_value_zscore",
            f"{self.asset}_{self.window}_avg_win_return",
            f"{self.asset}_{self.window}_avg_loss_return",
            f"{self.asset}_{self.window}_max_single_drawdown",
            # f"{self.asset}_{self.window}_avg_trade_size",
            f"{self.asset}_{self.window}_realized_pnl",
            f"{self.asset}_{self.window}_cum_return",
            # f"{self.asset}_{self.window}_episode_progress",
            # f"{self.asset}_{self.window}_sterling_ratio",
            f"{self.asset}_{self.window}_ulcer_ratio",
            f"{self.asset}_{self.window}_cdar",
            f"{self.asset}_{self.window}_avg_holding_period",
            f"{self.asset}_{self.window}_new_high_count",
            f"{self.asset}_{self.window}_recovery_ratio",
            f"{self.asset}_{self.window}_skew_vol_ratio",
            f"{self.asset}_{self.window}_mean_return",
            f"{self.asset}_{self.window}_variance",
            f"{self.asset}_{self.window}_range_return",
            f"{self.asset}_{self.window}_iqr_return",
            f"{self.asset}_{self.window}_jarque_bera",
            f"{self.asset}_{self.window}_autocorr_sq",
            f"{self.asset}_{self.window}_drawdown_rate",
            f"{self.asset}_{self.window}_drawdown_entropy",
            f"{self.asset}_{self.window}_win_loss_ratio",
            # f"{self.asset}_{self.window}_avg_win_streak",
            # f"{self.asset}_{self.window}_avg_loss_streak",
            f"{self.asset}_{self.window}_tail_return_ratio",
            f"{self.asset}_{self.window}_spectral_entropy",
            f"{self.asset}_{self.window}_avg_drawdown",
            f"{self.asset}_{self.window}_annual_return",
            f"{self.asset}_{self.window}_annual_volatility",
            f"{self.asset}_{self.window}_annual_sharpe",
            # f"{self.asset}_{self.window}_cagr",
            f"{self.asset}_{self.window}_trend_slope",
            f"{self.asset}_{self.window}_trend_r2",
            f"{self.asset}_{self.window}_return_consistency",
            f"{self.asset}_{self.window}_zero_crossing_rate",
            f'{self.asset}_{self.window}_hurst_exponent',
            ]
        
        self.metrics.update(
            {
                col: 0 for col in self.col_names
                }
            )
        
    def reset(self):
        self.asset_values.clear()
        self.market_returns.clear()
        self.risk_free_rates.clear()
        self.timestamps.clear()
        self.latest_features = {}
        self.returns.clear()

        self.portfolio_excess_returns.clear()
        self.benchmark_excess_returns.clear()
        
        # self._peak_value = None
        # self._peak_index = None
        # self._max_drawdown = 0.0
        # self._max_drawdown_length = 0
        # self._current_drawdown_start = None
        # self._current_drawdown_start: Optional[int] = None

        self.weights_history.clear()
        self.benchmark_returns.clear()
        self.drawdowns.clear()
        self.vol_history.clear()
        self.asset_values.clear()
        self.momentum_history.clear()
        self.momentum_history.clear()
        
        self.cash_ratios.clear() # : List[float] = []
        # self.asset_counts.clear() # : List[float] = []
        self.concentration_ratios.clear() # : List[float] = []
        self.trade_counts.clear() # : List[float] = []
        self.value_zscores.clear()
        self.realized_pnls.clear()
        self.cumulative_returns.clear()
        self.price_history.clear()

        self.metrics.update(
            {
                col: 0 for col in self.col_names
                }
            )

    def record(
        self, asset_value,
        market_return=0.0, risk_free_rate=None, timestamp=None
        ):
        self.asset_values.append(asset_value)
        self.market_returns.append(market_return)

        # self.portfolio_excess_returns.append(
        if risk_free_rate is None:
            if isinstance(self.static_risk_free, (float, int)):
                self.risk_free_rates.append(self.static_risk_free)
            else:
                self.risk_free_rates.append(
                    self.static_risk_free[len(self.asset_values) - 1]
                )
        else:
            self.risk_free_rates.append(risk_free_rate)
        self.timestamps.append(timestamp)

        risk_free = self.risk_free_rates[-1]
        values = np.array(self.asset_values)[:]

        port_rets = np.diff(values) / values[:-1] # if values.size > 1 else values[-1]
        excess_port = port_rets - risk_free
        excess_market = np.mean(self.market_returns) - risk_free

        self.benchmark_excess_returns.append(excess_market)
        self.portfolio_excess_returns.extend(excess_port)

        if len(self.asset_values) >= self.window:
            self.latest_features = self._compute_features()
        else:
            self.latest_features.update(
                {col: 0 for col in self.col_names}
                )

    def m_squared_ratio(self, risk_free_rate, debug=False):
        if len(self.asset_values) >= 3:
            
            portfolio_std = np.std(self.portfolio_excess_returns)
            benchmark_std = np.std(self.benchmark_excess_returns)

            if portfolio_std == 0 or benchmark_std == 0:
                return 0.

            elif portfolio_std > 1e-8 and benchmark_std > 1e-8:
                portfolio_mean = np.mean(self.portfolio_excess_returns)
                adjusted_portfolio_performance = risk_free_rate + (
                    (portfolio_mean - risk_free_rate) * (benchmark_std / portfolio_std)
                    )
                # return ( # Original.
                m_squared_ratio = float(
                    (adjusted_portfolio_performance - risk_free_rate) - (
                        np.mean(self.benchmark_excess_returns) - risk_free_rate
                        )
                    ) * (benchmark_std / portfolio_std)
                if debug:
                    print(
f'''
||| portfolio_mean: {portfolio_mean}
||| adjusted_portfolio_performance: {adjusted_portfolio_performance}
||| benchmark_std: {benchmark_std}
||| portfolio_std: {portfolio_std}
||| risk_free_rate: {risk_free_rate}
||| m_squared_ratio: {m_squared_ratio}
'''
                    )
                return m_squared_ratio
            else:
                return 0.0
        else:
            return 0.0

    def _compute_features(self, debug=False):
        values = np.array(self.asset_values)[-self.window:]
        market_ret = np.array(self.market_returns)[-self.window:]
        risk_free = np.array(self.risk_free_rates)[-self.window:]

        port_rets = np.diff(values) / values[:-1]
        port_rets = np.insert(port_rets, 0, 0.0) # 0 for first

        # Excess returns
        excess_port = port_rets - risk_free
        excess_market = market_ret - risk_free

        # Downside
        mean_excess = np.mean(excess_port)
        # std_excess = np.std(excess_port) + 1e-12
        # downside = np.std(np.minimum(0, excess_port)) + 1e-12
        if np.std(excess_port) > 1e-8:
            std_excess = np.std(excess_port) + 1e-12
        else:
            std_excess = 0.0
        if np.std(np.minimum(0, excess_port)) > 1e-8:
            downside = np.std(np.minimum(0, excess_port)) + 1e-12
        else:
            downside = 0.0

        # Alpha (annualized mean excess return)
        alpha_annual = mean_excess * 252 # 7

        # Volatility (annualized)
        vol_annual = np.std(port_rets) * np.sqrt(252) # 7

        # Skewness and Kurtosis
        skewness = skew(port_rets)
        kurt = kurtosis(port_rets)

        # CVaR (Conditional Value-at-Risk, 5%)
        q05 = np.percentile(port_rets, 5)
        cvar_05 = port_rets[
            port_rets <= q05].mean() if np.any(port_rets <= q05) else 0.0

        # Rolling Sharpe
        # sharpe = mean_excess / std_excess
        if std_excess > 1e-8:
            sharpe = mean_excess / std_excess
        else:
            sharpe = 0.0
        if downside > 1e-8:
            sortino = mean_excess / downside
        else:
            sortino = 0.0

        # Max Drawdown
        cumulative = np.cumprod(1 + port_rets)
        drawdowns = 1 - cumulative / np.maximum.accumulate(cumulative)
        max_drawdown = np.max(drawdowns)

        m_squared_ratio = self.m_squared_ratio(risk_free_rate=risk_free[-1], debug=debug)

        features = {
            # f"{self.asset}_{self.window}_alpha_annual": alpha_annual,
            # f"{self.asset}_{self.window}_vol_annual": vol_annual,
            f"{self.asset}_{self.window}_alpha_monthly": alpha_annual,
            f"{self.asset}_{self.window}_vol_monthly": vol_annual,
            # f"{self.asset}_{self.window}_skew": skewness,
            # f"{self.asset}_{self.window}_kurtosis": kurt,
            # f"{self.asset}_{self.window}_cvar_05": cvar_05,
            f"{self.asset}_{self.window}_sharpe": sharpe,
            f"{self.asset}_{self.window}_sortino": sortino,
            f"{self.asset}_{self.window}_max_drawdown": max_drawdown,
            f"{self.asset}_{self.window}_m_squared_ratio": m_squared_ratio,
        }

        if len(self.asset_values) >= self.window:
            if self.compute_market_beta:
                if np.std(excess_market) > 1e-8:
                    beta = np.cov(excess_port, excess_market)[0, 1] / np.var(excess_market)
                else:
                    beta = 0.0
                if beta > 1e-8:
                    basr = sharpe / (beta)
                else:
                    basr = 0.0

                features[f"{self.asset}_{self.window}_beta"] = beta
                features[f"{self.asset}_{self.window}_beta_adj_sharpe"] = basr
                if debug:
                    print(
f'''
||| np.std(excess_market): {np.std(excess_market)}
||| excess_port.mean(): {np.mean(excess_port)}
||| excess_market.std(): {np.std(excess_market)}
||| np.std(excess_market): {np.std(excess_market)}
||| np.cov(excess_port, excess_market)[0, 1]: {np.cov(excess_port, excess_market)[0, 1]}
||| np.var(excess_market): {np.var(excess_market)}
||| beta_adj_sharpe: {basr}
'''
                    )
        else:
            features[f"{self.asset}_{self.window}_beta"] = 0
            features[f"{self.asset}_{self.window}_beta_adj_sharpe"] = 0
        return features

    def get_latest_features(self):
        return self.latest_features.copy()


    def as_dataframe(self):
        feats = []
        vals = np.array(self.asset_values)
        market = np.array(self.market_returns)
        risk_free = np.array(self.risk_free_rates)
        ts = np.array(self.timestamps)
        for i in range(self.window, len(vals) + 1):
            v = vals[i - self.window:i]
            m = market[i - self.window:i]
            r = risk_free[i - self.window:i]
            t = ts[i - 1]  # last timestamp in window

            port_rets = np.diff(v) / v[:-1]
            port_rets = np.insert(port_rets, 0, 0.0)
            excess_port = port_rets - r
            excess_market = m - r

            mean_excess = np.mean(excess_port)
            std_excess = np.std(excess_port) # + 1e-12
            downside = np.std(np.minimum(0, excess_port)) # + 1e-12
            alpha_annual = mean_excess * 252 # 7 # 252
            vol_annual = np.std(port_rets) * np.sqrt(252) # 7) # 252)
            skewness = skew(port_rets)
            kurt = kurtosis(port_rets)
            q05 = np.percentile(port_rets, 5)
            cvar_05 = port_rets[port_rets <= q05].mean() if np.any(port_rets <= q05) else 0.0
            if std_excess > 1e-12:
                sharpe = mean_excess / std_excess
            else:
                sharpe = 0.
            if downside > 1e-12:
                sortino = mean_excess / downside
            else:
                sortino = 0.
            cumulative = np.cumprod(1 + port_rets)
            drawdowns = 1 - cumulative / np.maximum.accumulate(cumulative)
            max_drawdown = np.max(drawdowns)

            f = {
                "timestamp": t,
                f"{self.asset}_{self.window}_alpha_annual": alpha_annual,
                f"{self.asset}_{self.window}_vol_annual": vol_annual,
                f"{self.asset}_{self.window}_skew": skewness,
                f"{self.asset}_{self.window}_kurtosis": kurt,
                f"{self.asset}_{self.window}_cvar_05": cvar_05,
                f"{self.asset}_{self.window}_sharpe": sharpe,
                f"{self.asset}_{self.window}_sortino": sortino,
                f"{self.asset}_{self.window}_max_drawdown": max_drawdown,
            }
            if self.compute_market_beta:
                if np.std(excess_market) > 1e-8:
                    beta = np.cov(
                        excess_port, excess_market
                        )[0, 1] / np.var(
                            excess_market
                            )
                else:
                    beta = 0.0
                basr = sharpe / (beta + 1e-12)
                f[f"{self.asset}_{self.window}_beta"] = beta
                f[f"{self.asset}_{self.window}_beta_adj_sharpe"] = basr
            feats.append(f)
        return pd.DataFrame(feats)

    @staticmethod
    def hurst_exponent(r: np.ndarray) -> float:
        """Estimate Hurst exponent via R/S analysis for 0 < H < 1."""
        N = len(r)
        if N < 2:
            return 0.5
        mean_r = np.mean(r)
        Y = np.cumsum(r - mean_r)
        R = np.max(Y) - np.min(Y)
        S = np.std(r, ddof=1) if np.std(r, ddof=1) > 1e-8 else 0.
        if S == 0 or R == 0:
            return 0.5
        return float(np.log(R / S) / np.log(N))

    def update(
        self,
        asset_value: float,
        # weights: np.ndarray,
        current_price: float,
        previous_price: float,
        previous_weight: float,
        current_weight: float,
        prev_asset_quantity: float,
        prev_balance: float,
        weight: float,
        cash: float,
        benchmark_return: Optional[float] = None,
        debug: bool = False,
    ):
        """
        Call this method once per step to record the latest portfolio state.
        
        :param asset_value: Current total portfolio value.
        :param weights: Current portfolio weights vector (must sum to 1).
        :param benchmark_return: Optional single-step benchmark return.
        """
        # Initialize on first call
        if not self.asset_values: # Original but not working.
        # if len(self.asset_values) == 0:
        # if self.current_step == self.start_step:
            self._peak_value = asset_value
            self._peak_index = 0
            self._current_drawdown_start = 0
            self.asset_values.append(asset_value)
            # self.weights_history.append(weights.copy())
            self.weights_history.append(weight)
            if benchmark_return is not None:
                self.benchmark_returns.append(benchmark_return)
            # else:
            #     self.benchmark_returns.append(0.)
            self.drawdowns.append(0.0)
            self.vol_history.append(0.0)
            self.returns.append(0.0)
            self.prev_ema_short = asset_value
            self.prev_ema_long = asset_value
            return # Original.
        
        # Compute and store return
        # prev_value = self.asset_values[-1] # Original.
        # ret = asset_value / prev_value - 1 # Original.
        # print(f'||| current_price: {current_price} ||| previous_price: {previous_price} |||')

        delta = float(
            float(current_price - previous_price) / previous_price
            ) if previous_price != 0. else 0.

        # prev_asset_quantity_list = []
        # for asset in self.selected_assets:
        #     prev_asset_quantity_list.append(
        #         float(
        #             self.historical_trades.at[
        #                 self.previous_timestamp, f'portfolio_{asset}_owned'
        #                 ]
        #             )
        #         )

        # should_be_asset_value = prev_balance + sum(
        #     [
        #         p_q * p_p for p_q, p_p in zip(
        #             prev_asset_quantity_list, previous_prices
        #             )
        #         ]
        #     )
        should_be_asset_value = prev_balance + float(
            # prev_asset_quantity * previous_price
            prev_asset_quantity * current_price
            )
        ret = (self.asset_value - should_be_asset_value) / should_be_asset_value if should_be_asset_value != 0. else 0.
        # ret = float(
        #     self.asset_value - should_be_asset_value
        #     ) / self.previous_asset_value if self.previous_asset_value != 0. else 0.
        # ret = float(
        #     float(
        #         asset_value - should_be_asset_value
        #         ) / self.prev_asset_value if len(self.returns) > 1 and self.prev_asset_value != 0. else float(float(asset_value - should_be_asset_value) / should_be_asset_value)
        #     )
        self.asset_value = asset_value
        self.returns.append(float(ret)) # Original.
        self.prev_asset_value = self.returns[-2] if len(self.returns) > 1 else self.returns[-1]
        self.should_be_asset_values.append(should_be_asset_value)
        
        if len(self.returns) > self.window:
            self.returns.pop(0)
        
        # Store benchmark return
        if benchmark_return is not None:
            self.benchmark_returns.append(benchmark_return)
            if len(self.benchmark_returns) > self.window:
                self.benchmark_returns.pop(0)
        # else:
        #     self.benchmark_returns.append(0.)
        
        # Update portfolio value history
        self.asset_values.append(asset_value) # Not Already happened in the step function.
        self.price_history.append(current_price)
        
        if len(self.price_history) > self.momentum_window + 1:
        # if len(self.asset_values) > self.momentum_window + 1:
            self.asset_values.pop(0)
        # Compute and store price momentum
        if len(self.price_history) > self.momentum_window:
        # if len(self.asset_values) > self.momentum_window:
            entry_price = self.asset_values[-(self.momentum_window + 1)]
            '''if entry_price > 0:
                if initial > 1:
                    price_mom = asset_value / entry_price - 1 if entry_price != 0. else 0.
            elif initial < 0:
                price_mom = asset_value / entry_price - 1 if entry_price != 0. else 0.
            else:
                price_mom = 0.'''

            price_mom = asset_value / entry_price - 1 if entry_price != 0. else 0.
        else:
            price_mom = 0.0
        self.momentum_history.append(price_mom)
        if len(self.momentum_history) > self.window:
            self.momentum_history.pop(0)
        
        # Update drawdown tracking
        if asset_value > self._peak_value:
            self._peak_value = asset_value
            self._peak_index = len(self.asset_values) - 1
            self._current_drawdown_start = self._peak_index
        else:
            drawdown = (
                self._peak_value - asset_value
                ) / self._peak_value if self._peak_value != 0. else 0.
            dd_length = len(
                self.asset_values
                ) - self._current_drawdown_start

            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
            # self._max_drawdown = max(self._max_drawdown, drawdown)

            if dd_length > self._max_drawdown_length:
                self._max_drawdown_length = dd_length
            # self._max_drawdown_length = max(self._max_drawdown_length, dd_length)

        self.drawdowns.append(self._max_drawdown)
        if len(self.drawdowns) > self.window:
            self.drawdowns.pop(0)


        # Compute rolling volatility (std of returns)
        vol = float(
            np.std(self.returns)
            ) if len(self.returns) > 0 and np.std(self.returns) > 1e-8 else 0.0
        self.vol_history.append(vol)
        if len(self.vol_history) > self.window:
            self.vol_history.pop(0)
        
        # Store weights for turnover calculation
        self.weights_history.append(current_weight)
        if len(self.weights_history) > self.window + 1:
            self.weights_history.pop(0)

        # Update EMAs
        self.prev_ema_short = (
            self.alpha_short * asset_value +
            (1 - self.alpha_short) * self.prev_ema_short
        )
        self.prev_ema_long = (
            self.alpha_long * asset_value +
            (1 - self.alpha_long) * self.prev_ema_long
        )
        if debug:
            print(
f'''
||| asset_value: {asset_value}
||| self.alpha_long: {self.alpha_long}
||| self.prev_ema_long: {self.prev_ema_long}
||| self.alpha_short: {self.alpha_short}
||| self.prev_ema_short: {self.prev_ema_short}
'''
            )
            # stop

        # 1. Cash ratio
        # cash = float(self.portfolio.get('cash', 0.0))
        if self.asset != 'cash':
            cash_ratio = cash / asset_value if asset_value != 0 else 0.0
            self.cash_ratios.append(cash_ratio)
            if len(self.cash_ratios) > self.window:
                self.cash_ratios.pop(0)

        # 2. Asset count (non-zero holdings excluding cash)
        # asset_count = float(
        #     sum(
        #         1 for k, v in self.portfolio.items() if k != 'cash' and v > 0
        #         )
        #     )
        # self.asset_counts.append(asset_count)
        # if len(self.asset_counts) > self.window:
        #     self.asset_counts.pop(0)

        # 3. Concentration ratio (sum of two largest weights)
        # sorted_w = np.sort(weight)[::-1]
        # conc_ratio = float(np.sum(sorted_w[:2]))
        # self.concentration_ratios.append(conc_ratio)
        # if len(self.concentration_ratios) > self.window:
        #     self.concentration_ratios.pop(0)

        # 4. Trade count: did a trade occur this step?
        if len(self.weights_history) > 1:
            prev_w = self.weights_history[-2]
            diff = float(
                np.sum(
                    np.abs(weight - prev_w)
                    )
                )
            # traded = 1.0 if diff > self.min_trade_value else 0.0
            traded = 1.0 if diff != 0. else 0.0
        else:
            traded = 0.0
        # self.trade_counts.append(self.traded)
        self.trade_counts.append(traded)
        if len(self.trade_counts) > self.window:
            self.trade_counts.pop(0)

        # print(f'||| self.window: {self.window} ||| self.asset_values: {self.asset_values} |||')
        # cast the deque to a NumPy array, then take the last `window` elements
        arr = np.array(self.asset_values, dtype=np.float32)
        # arr_list = list(self.asset_values)
        # 5. Portfolio value Z-score over recent window
        vals = np.array(
            arr[int(-self.window):], dtype=np.float32
            ) # if len(self.asset_values) > 1 else np.array(self.asset_values[-1:], dtype=np.float32)
        if vals.size > 1:
            mean_v = vals.mean()
            std_v = vals.std() if len(self.returns) > 0 and vals.std() > 1e-8 else 0.0
            vz = float(
                (asset_value - mean_v) / std_v if std_v > 1e-8 else 0.
                ) if std_v > 0 else 0.0
        else:
            vz = 0.0
        self.value_zscores.append(vz)
        if len(self.value_zscores) > self.window:
            self.value_zscores.pop(0)

        # Realized PnL
        pv_list = list(self.asset_values)
        if len(pv_list) > 1:
            pnl = pv_list[-1] - pv_list[-2]
        else:
            pnl = 0.0
            
        self.realized_pnls.append(pnl)
        if len(self.realized_pnls) > self.window:
            self.realized_pnls.pop(0)

        # Cumulative return
        initial = pv_list[0] if pv_list else asset_value
        '''if initial > 0:
            if initial > 1:
                cum_ret = (asset_value / initial - 1)
        elif initial < 0:
            cum_ret = (asset_value / initial - 1)
        else: 
            cum_ret = 0.'''
        cum_ret = (asset_value / initial - 1) if initial != 0. else 0.
        self.cumulative_returns.append(cum_ret)
        if len(self.cumulative_returns) > self.window:
            self.cumulative_returns.pop(0)



    def compute_metrics(self, debug=False) -> Dict[str, float]:
        """
        Compute and return all configured financial metrics.
        """
        metrics: Dict[str, float] = {}
        r = np.array(self.returns)
        
        # Basic stats
        mean_r = r.mean() if r.size > 0 else 0.0
        std_r = r.std() if r.size > 0 and r.std() > 1e-8 else 0.0
        
        # Sharpe ratio
        excess = r - self.risk_free_rate_
        metrics[f'{self.asset}_{self.window}_sharpe'] = (excess.mean() / std_r) if std_r > 0 else 0.0
        
        # Sortino ratio
        neg_r = r[r < self.risk_free_rate_]
        downside_std = neg_r.std() if neg_r.size > 0 and neg_r.std() > 1e-8 else 0.0
        metrics[f'{self.asset}_{self.window}_sortino'] = float(
            excess.mean() / downside_std) if downside_std > 0 else 0.0

        # Momentum (cumulative return)
        metrics[f'{self.asset}_{self.window}_momentum'] = float(np.prod(1 + r) - 1) if r.size > 0 else 0.0

        # Rolling skewness & kurtosis
        if std_r > 0:
            skewness = float(np.mean((r - mean_r)**3) / (std_r**3))
            kurtosis = float(np.mean((r - mean_r)**4) / (std_r**4) - 3.0)
            metrics[f'{self.asset}_{self.window}_skewness'] = skewness
            metrics[f'{self.asset}_{self.window}_kurtosis'] = kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
            metrics[f'{self.asset}_{self.window}_skewness'] = 0.0
            metrics[f'{self.asset}_{self.window}_kurtosis'] = 0.0

        # Realized volatility annualized (assumes daily steps)
        realized_vol = std_r * np.sqrt(252) if std_r > 0 else 0.0
        metrics[f'{self.asset}_{self.window}_realized_vol'] = realized_vol

        # Ulcer Index: sqrt of mean squared drawdowns
        dd_arr = np.array(self.drawdowns)
        ulcer_index = float(np.sqrt(np.mean(dd_arr**2))) if dd_arr.size > 0 or np.mean(dd_arr**2) > 0. else 0.0
        metrics[f'{self.asset}_{self.window}_ulcer_index'] = ulcer_index

        # Volatility of volatility
        vh = np.array(self.vol_history)
        metrics[f'{self.asset}_{self.window}_vol_of_vol'] = float(np.std(vh)) if vh.size > 0 and np.std(vh) > 1e-8 else 0.0

        # Beta & correlation with benchmark if available
        """if self.benchmark_returns:
            br = np.array(self.benchmark_returns)
            if debug:
                print(
f'''
||| self.benchmark_returns: {self.benchmark_returns}
||| r: {r}
||| len(r): {len(r)}
||| br: {br}
||| len(br): {len(br)}
'''
                 )
            cov = float(np.cov(r, br, ddof=0)[0, 1]) if br.size >=3 else 0.0
            var_b = float(np.var(br)) if br.size > 0 and np.var(br) > 1e-8 else 0.0
            metrics[f'{self.asset}_{self.window}_beta'] = cov / var_b if var_b > 0 else 0.0
            correlation = float(np.corrcoef(r, br)[0, 1]) if br.size > 1 else 0.0
            if not np.isnan(correlation) and not np.isinf(correlation):
                metrics[f'{self.asset}_{self.window}_correlation'] = correlation if br.size > 1 else 0.0 # float(np.corrcoef(r, br)[0, 1]) if br.size > 1 else 0.0
            else:
                metrics[f'{self.asset}_{self.window}_correlation'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_beta'] = 0.0
            metrics[f'{self.asset}_{self.window}_correlation'] = 0.0"""


        # Beta & correlation with benchmark if available
        if self.benchmark_returns:
            r = np.asarray(self.returns, dtype=float)
            br = np.asarray(self.benchmark_returns, dtype=float)

            # 1) covariance and variance via safe_cov
            cov_rb = safe_cov(r, br, ddof=0)
            var_b = safe_cov(br, ddof=0)


            if debug:
                print(
f'''
||| self.benchmark_returns: {self.benchmark_returns}
||| r: {r}
||| len(r): {len(r)}
||| br: {br}
||| len(br): {len(br)}
'''
                 )

            # 2) beta
            metrics[f'{self.asset}_{self.window}_beta'] = cov_rb / var_b if var_b != 0.0 else 0.0

            try:
                correlation = float(np.corrcoef(r, br)[0, 1]) if br.size > 1 else 0.0
            except RuntimeWarning as e:
                print("Caught invalid‐divide warning:", e)
                # fallback to safe routine
                correlation = float(safe_corrcoef(r, br)[0, 1])

            metrics[f'{self.asset}_{self.window}_correlation'] = correlation

            # 3) correlation
            var_r = safe_cov(r, ddof=0)
            # if var_r > 0.0 and var_b > 0.0:
            #     metrics[f'{self.asset}_{self.window}_correlation'] = cov_rb / (math.sqrt(var_r) * math.sqrt(var_b))
            #     metrics[f'{self.asset}_{self.window}_correlation'] = correlation
            # else:
            #     metrics[f'{self.asset}_{self.window}_correlation'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_correlation'] = 0.0
            metrics[f'{self.asset}_{self.window}_beta'] = 0.0
        if debug:
            print(
f'''
||| self.benchmark_returns: {self.benchmark_returns}
||| r: {r}
||| correlation: {correlation}
||| cov_rb / var_b | beta: {cov_rb / var_b}
||| cov_rb: {cov_rb}
||| var_b: {var_b}
'''
             )
        
        # Omega ratio (threshold at risk-free rate)
        gains = r[r >= self.risk_free_rate_] - self.risk_free_rate_
        losses = self.risk_free_rate_ - r[r < self.risk_free_rate_]
        sum_gains = gains.sum() if gains.size > 0 else 0.0
        sum_losses = losses.sum() if losses.size > 0 else 1.0
        metrics[f'{self.asset}_{self.window}_omega'] = sum_gains / sum_losses
        
        # Calmar ratio
        metrics[f'{self.asset}_{self.window}_calmar_mean_r'] = (mean_r / self._max_drawdown) if self._max_drawdown != 0 else 0.0
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        if r.size > 0:
            var_level = np.percentile(r, 100 * 0.05) # self.var_percentile)
            metrics[f'{self.asset}_{self.window}_var'] = -var_level
            metrics[f'{self.asset}_{self.window}_cvar'] = -r[r <= var_level].mean() if r[r <= var_level].size > 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_var'] = 0.0
            metrics[f'{self.asset}_{self.window}_cvar'] = 0.0
        
        # Maximum Drawdown (depth) and Length
        metrics[f'{self.asset}_{self.window}_max_drawdown'] = self._max_drawdown
        metrics[f'{self.asset}_{self.window}_max_drawdown_length'] = float(
            self._max_drawdown_length
            )
        
        # Turnover and Transaction Cost
        # if len(self.weights_history) > 1:
        #     prev_w = self.weights_history[-2]
        #     curr_w = self.weights_history[-1]
        #     turnover = float(np.abs(curr_w - prev_w).sum())
        #     # cost = turnover * self.transaction_cost_rate
        #     cost = turnover * self.total_transaction_fee
        # else:
        #     turnover, cost = 0.0, 0.0
        # metrics[f'{self.asset}_{self.window}_turnover'] = turnover
        # metrics[f'{self.asset}_{self.window}_transaction_cost'] = cost
        
        # Entropy bonus for diversification
        w = np.clip(self.weights_history[-1], 1e-8, 1.0)
        w /= w.sum()
        # metrics[f'{self.asset}_{self.window}_entropy'] = -float(np.sum(w * np.log(w)))
        
        # Risk-parity deviation (distance from equal-weight)
        n = w.size
        target = np.ones(n) / n
        # metrics[f'{self.asset}_{self.window}_risk_parity_deviation'] = float(
        #     np.linalg.norm(w - target)
        #     )
        
        # Benchmark-relative and Information Ratio
        if self.benchmark_returns:
            br = np.array(self.benchmark_returns)
            active = r - br
            trk_err = active.std() if active.size > 0 and active.std() > 1e-8 else 0.0 # tracking error.
            # info_std = float(active.std()) if active.size > 0 else 0.0
            metrics[f'{self.asset}_{self.window}_excess_return'] = float(mean_r - br.mean())
            metrics[f'{self.asset}_{self.window}_tracking_error'] = float(trk_err)
            metrics[f'{self.asset}_{self.window}_information_ratio'] = float(
                metrics[f'{self.asset}_{self.window}_excess_return'] / trk_err
                ) if trk_err > 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_excess_return'] = 0.0
            metrics[f'{self.asset}_{self.window}_tracking_error'] = 0.0
            metrics[f'{self.asset}_{self.window}_information_ratio'] = 0.0

        # EMA & MACD
        metrics[f'{self.asset}_{self.window}_ema_short'] = float(self.prev_ema_short)
        metrics[f'{self.asset}_{self.window}_ema_long'] = float(self.prev_ema_long)
        # metrics[f'{self.asset}_{self.window}_macd'] = metrics[f'{self.asset}_{self.window}_ema_short'] - metrics[f'{self.asset}_{self.window}_ema_long']

        # RSI
        if len(self.returns) >= self.rsi_window:
            recent = r[-self.rsi_window:]
            gains = recent[recent > 0]
            losses = -recent[recent < 0]
            avg_gain = float(np.mean(gains)) if gains.size > 0 else 0.0
            avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
            metrics[f'{self.asset}_{self.window}_rsi'] = float(100 * avg_gain / (avg_gain + avg_loss)) if (avg_gain + avg_loss) != 0 else 50.0
        else:
            metrics[f'{self.asset}_{self.window}_rsi'] = 50.0

        # Win rate & Profit factor
        metrics[f'{self.asset}_{self.window}_win_rate'] = float((r > 0).sum() / r.size) if r.size > 0 else 0.0
        pos = float(np.sum(r[r > 0])) if r.size > 0 else 0.0
        neg = float(-np.sum(r[r < 0])) if r.size > 0 else 0.0
        metrics[f'{self.asset}_{self.window}_profit_factor'] = float(pos / neg) if neg > 0 else 0 # np.inf

        # Rachev ratio
        if r.size > 0:
            # k = max(int(self.rachev_percentile * r.size), 1)
            k = max(int(0.05 * r.size), 1)
            
            sorted_r = np.sort(r)
            tail_up = sorted_r[-k:]
            tail_down = sorted_r[:k]
            metrics[f'{self.asset}_{self.window}_rachev'] = float(
                np.mean(tail_up) / -np.mean(tail_down)
                ) if np.mean(tail_down) != 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_rachev'] = 0.0

        # Calmar ratio
        cum_return = float(np.prod(1 + r) - 1) if r.size > 0 else 0.0
        max_dd = float(np.max(dd_arr)) if dd_arr.size > 0 else 0.0
        metrics[f'{self.asset}_{self.window}_calmar_cum_r'] = float(cum_return / max_dd) if max_dd != 0 else 0.0

        # Drawdown length (consecutive)
        # length = 0
        # Drawdown recovery time: consecutive steps since last drawdown cleared
        recovery = 0
        dd = np.array(self.drawdowns, dtype=float)
        for d in dd_arr[::-1]:
            if d > 0:
                recovery += 1
            else:
                break
        # metrics[f'{self.asset}_{self.window}_drawdown_length_consecutive'] = float(length)
        metrics[f'{self.asset}_{self.window}_drawdown_recovery_time'] = float(recovery)

        # Autocorrelation lag 1
        if r.size > 1:
            autocorr_lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            if not np.isnan(autocorr_lag1) and not np.isinf(autocorr_lag1):
                metrics[f'{self.asset}_{self.window}_autocorr_lag1'] = autocorr_lag1 # float(np.corrcoef(r[:-1], r[1:])[0, 1])
            else:
                metrics[f'{self.asset}_{self.window}_autocorr_lag1'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_autocorr_lag1'] = 0.0

        # Hurst exponent
        metrics[f'{self.asset}_{self.window}_hurst'] = self.hurst_exponent(r)

        # Efficiency ratio: net change / sum of absolute changes
        if r.size > 0:
            net = abs(np.sum(r))
            tot = np.sum(np.abs(r))
            metrics[f'{self.asset}_{self.window}_efficiency_ratio'] = float(net / tot) if tot != 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_efficiency_ratio'] = 0.0

        # Kelly fraction: mean / variance
        mean_r = float(np.mean(r)) if r.size > 0 else 0.0
        var_r = float(np.var(r)) if r.size > 0 and np.var(r) > 1e-8 else 1.0 # 0.0 # 1.0
        # print(f'||| mean_r: {mean_r} ||| var_r: {var_r} ||| r: {r} |||')
        metrics[f'{self.asset}_{self.window}_kelly_fraction'] = float(mean_r / var_r) if var_r > 1e-8 else float(0.)

        # Last return Z-score
        if r.size > 1:
            if r.size > 0:
                metrics[f'{self.asset}_{self.window}_last_return_z'] = float(
                    (r[-1] - mean_r) / (
                        np.std(r) if np.std(r) > 1e-8 else 1
                        )
                    )
            else:
            #  if len(self.returns) >= 3 else 0.0
                metrics[f'{self.asset}_{self.window}_last_return_z'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_last_return_z'] = 0.0

        arr = np.array(self.asset_values)
        # Price Z-score relative to rolling price_window
        prices = np.array(
            arr[
                -(self.window+1):
                # -(self.price_window+1):
                ], dtype=float
            )
        if prices.size > self.window:
            past = prices[:-1]
            mu = past.mean()
            if past.size > 0:
                sigma = past.std() if past.std() > 1e-8 else 0.
            else:
                sigma = 0.0
            metrics[f'{self.asset}_{self.window}_price_zscore'] = float(
                (prices[-1] - mu) / sigma
                ) if sigma > 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_price_zscore'] = 0.0


        # Compute MFE and MAE over the window
        if r.size > 0:
            cumrets = np.cumprod(1 + r) - 1
            metrics[f'{self.asset}_{self.window}_mfe'] = float(np.max(cumrets))
            metrics[f'{self.asset}_{self.window}_mae'] = float(np.min(cumrets))
            # Tail risk: variance and downside volatility of the worst q%
            # var_lvl = np.percentile(r, 100 * self.var_percentile)
            var_lvl = np.percentile(r, 100 * 0.05)
            tail = r[r <= var_lvl]
            # metrics[f'{self.asset}_{self.window}_tail_variance'] = float(
            #     np.var(tail)) if tail.size > 0 and np.var(tail) > 1e-8 else 0.0
            # metrics[f'{self.asset}_{self.window}_downside_vol'] = float(
            #     np.std(tail)) if tail.size > 0 and np.std(tail) > 1e-8 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_mfe'] = 0.0
            metrics[f'{self.asset}_{self.window}_mae'] = 0.0
            # metrics[f'{self.asset}_{self.window}_tail_variance'] = 0.0
            # metrics[f'{self.asset}_{self.window}_downside_vol'] = 0.0

        # Compute price momentum z-score
        mh = np.array(self.momentum_history, dtype=float)
        if mh.size > 1:
            base = mh[:-1]
            mean_m = np.mean(base)
            if base.size > 0:
                std_m = np.std(base) if np.std(base) > 1e-8 else 0
            else:
                std_m = 0.0
            metrics[f'{self.asset}_{self.window}_momentum_zscore'] = float(
                (mh[-1] - mean_m) / std_m
                ) if std_m > 1e-8 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_momentum_zscore'] = 0.0

        # Return distribution entropy
        if r.size > 1:
            hist, _ = np.histogram(r, bins=self.entropy_bins, density=True)
            probs = hist / np.sum(hist + 1e-8)
            probs = probs[probs > 0]
            metrics[f'{self.asset}_{self.window}_return_entropy'] = -float(np.sum(probs * np.log(probs)))
        else:
            metrics[f'{self.asset}_{self.window}_return_entropy'] = 0.0

        # Variance ratio: var of k-step aggregated returns / var of single-step returns
        k = self.var_ratio_lag
        if r.size > k:
            if r.size > 0:
                agg = np.array([np.sum(r[i:i+k]) for i in range(len(r) - k + 1)])
                var_single = float(np.var(r)) if np.var(r) > 1e-8 else 0.
                metrics[f'{self.asset}_{self.window}_variance_ratio'] = float(np.var(agg) / var_single) if var_single > 0. else 0.
            else:
                metrics[f'{self.asset}_{self.window}_variance_ratio'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_variance_ratio'] = 0.0

        # Current win and loss streaks
        win_streak = 0
        for ret in r[::-1]:
            if ret > 0:
                win_streak += 1
            else:
                break
        loss_streak = 0
        for ret in r[::-1]:
            if ret < 0:
                loss_streak += 1
            else:
                break
        # metrics[f'{self.asset}_{self.window}_win_streak'] = float(win_streak)
        # metrics[f'{self.asset}_{self.window}_loss_streak'] = float(loss_streak)

        w = np.array(self.weights_history[-1], dtype=float)

        # if self.asset != 'cash':
        #     # 1. Effective number of bets (ENB)
        #     metrics[f'{self.asset}_{self.window}_effective_bets'] = float(
        #         1.0 / np.sum(w**2)) if w.size > 0 else 0.0

        # 2. Mean absolute deviation (MAD)
        if r.size > 0:
            mean_r = np.mean(r)
            metrics[f'{self.asset}_{self.window}_mad'] = float(
                np.mean(np.abs(r - mean_r)))
        else:
            metrics[f'{self.asset}_{self.window}_mad'] = 0.0

        # 3. Gini coefficient of returns (shifted to positive)
        if r.size > 0:
            r_shift = r - np.min(r) + 1e-8
            sorted_r = np.sort(r_shift)
            n = sorted_r.size
            index = np.arange(1, n+1)
            gini = (
                2 * np.sum(index * sorted_r) / (n * np.sum(sorted_r))
                ) - ((n+1) / n)
            metrics[f'{self.asset}_{self.window}_gini'] = float(gini)
        else:
            metrics[f'{self.asset}_{self.window}_gini'] = 0.0

        '''# 4. Hill estimator for tail index (negative returns tail)
        if r.size > 1:
            var_lvl = np.percentile(r, 100 * 0.05) # self.var_percentile)
            tail = -r[r < var_lvl]  # convert to positive losses
            if tail.size > 1:
                sorted_tail = np.sort(tail)[::-1]
                k = sorted_tail.size
                x_k = sorted_tail[-1]
                # print(f'||| x_k: {x_k} ||| sorted_tail: {sorted_tail} ||| k: {k} ||| tail: {tail} ||| var_lvl: {var_lvl} ||| r: {r} |||')
                hill = float(np.mean(np.log(sorted_tail / x_k))) # if x_k > 0. and all(sorted_tail) > 0. else float(np.mean(np.log(np.abs(sorted_tail) / np.abs(x_k))))
                metrics[f'{self.asset}_{self.window}_hill_estimator'] = hill
            else:
                metrics[f'{self.asset}_{self.window}_hill_estimator'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_hill_estimator'] = 0.0'''

        # 5. Drawdown at Risk (DaR)
        if dd.size > 0:
            metrics[f'{self.asset}_{self.window}_drawdown_at_risk'] = float(
                np.percentile(
                    dd, 100 * (1 - 0.05) # self.var_percentile)
                    )
                )
        else:
            metrics[f'{self.asset}_{self.window}_drawdown_at_risk'] = 0.0

        # 6. Coefficient of Variation (CoV)
        if r.size > 0:
            mean_r = np.mean(r)
            std_r = np.std(r) if np.std(r) > 1e-8 else 0.
            metrics[f'{self.asset}_{self.window}_cov'] = float(std_r / abs(mean_r)) if mean_r != 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_cov'] = 0.0

        # Latest values for env-specific features
        if self.asset != 'cash':
            metrics[f'{self.asset}_{self.window}_cash_ratio'] = self.cash_ratios[-1] if self.cash_ratios else 0.0
        # metrics[f'{self.asset}_{self.window}_asset_count'] = self.asset_counts[-1] if self.asset_counts else 0.0
        # metrics[f'{self.asset}_{self.window}_concentration_ratio'] = self.concentration_ratios[-1] if self.concentration_ratios else 0.0
        # metrics[f'{self.asset}_{self.window}_trade_count'] = sum(self.trade_counts)  # total trades in window
        metrics[f'{self.asset}_{self.window}_asset_value_zscore'] = self.value_zscores[-1] if self.value_zscores else 0.0
        
        # Average win and loss returns
        wins = r[r > 0]
        losses = -r[r < 0]
        metrics[f'{self.asset}_{self.window}_avg_win_return'] = float(wins.mean()) if wins.size > 0 else 0.0
        metrics[f'{self.asset}_{self.window}_avg_loss_return'] = float(losses.mean()) if losses.size > 0 else 0.0

        list_asset_values = list(self.asset_values)
        # Max single-step drawdown in window
        pv = np.array(list_asset_values[-(self.window+1):], dtype=float)
        if pv.size > 0:
            single_dd = (pv[:-1] - pv[1:]) / (pv[:-1] + 1e-8)
            metrics[f'{self.asset}_{self.window}_max_single_drawdown'] = float(np.max(single_dd)) if single_dd.size > 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_max_single_drawdown'] = 0.0

        # Avg trade size: turnover per trade
        trade_counts = np.array(self.trade_counts if hasattr(self, 'trade_counts') else [], dtype=np.float32)
        turnover_vals = np.array(self.concentration_ratios, dtype=float)  # reuse concentration buffer temporarily

        # Actually, trade_counts and turnover are in parent metrics, so compute here properly:
        # turnover = metrics.get('trade_count', 0.0)  # number of trades
        # total_turnover = metrics.get('turnover', 0.0)  # sum of weight changes
        total_trade_count = np.sum(trade_counts)
        # metrics[f'{self.asset}_{self.window}_avg_trade_size'] = float(turnover / total_trade_count) if total_trade_count > 0 else 0.0

        # Realized PnL metrics
        rp = np.array(self.realized_pnls, dtype=float)
        metrics[f'{self.asset}_{self.window}_realized_pnl'] = float(rp[-1]) if rp.size > 0 else 0.0
        cum_return = float(self.cumulative_returns[-1]) if self.cumulative_returns else 0.0
        metrics[f'{self.asset}_{self.window}_cum_return'] = cum_return

        # Episode progress
        # if self.horizon:
        #     step = len(self.asset_values) - 1
        #     metrics[f'{self.asset}_{self.window}_episode_progress'] = float(step / self.horizon)
        # else:
        #     metrics[f'{self.asset}_{self.window}_episode_progress'] = 0.0

        # Omega ratio: gains/losses at threshold = risk_free_rate
        # gains = r[r > self.risk_free_rate_] - self.risk_free_rate_
        # losses = self.risk_free_rate_ - r[r < self.risk_free_rate_]
        # metrics[f'{self.asset}_{self.window}_omega'] = float(np.sum(gains) / np.sum(losses)) if losses.size > 0 else float('inf')

        # Sterling ratio: annualized return / avg max drawdown (approx)
        # ann_return = np.mean(r) * 252 if r.size > 0 else 0.0
        # avg_dd = np.mean(dd) if dd.size > 0 else 0.0
        # metrics[f'{self.asset}_{self.window}_sterling_ratio'] = float(ann_return / avg_dd) if avg_dd != 0 else 0.0

        # Ulcer ratio: ulcer_index / cumulative return
        # cum_ret = metrics.get('cum_return', 0.0)
        # ui = metrics.get('ulcer_index', 0.0)
        metrics[f'{self.asset}_{self.window}_ulcer_ratio'] = float(ulcer_index / abs(cum_return)) if cum_return != 0 else 0.0

        # Conditional Drawdown at Risk (CDaR): average of worst q% drawdowns
        if dd.size > 0:
            # q = int(np.ceil(self.var_percentile * dd.size))
            q = int(np.ceil(0.05 * dd.size))
            worst_dd = np.sort(dd)[-q:]
            metrics[f'{self.asset}_{self.window}_cdar'] = float(np.mean(worst_dd))
        else:
            metrics[f'{self.asset}_{self.window}_cdar'] = 0.0

        # Average holding period: window / number of trades
        # trade_count = metrics.get('trade_count', 0.0)
        trade_count = sum(self.trade_counts)
        metrics[f'{self.asset}_{self.window}_avg_holding_period'] = float(self.window / trade_count) if trade_count != 0 else float(self.window)

        # New high count: times current value equals new rolling max within window
        pv = np.array(arr[-(self.window+1):], dtype=float)
        new_highs = np.sum(pv[1:] >= np.maximum.accumulate(pv[:-1]))
        metrics[f'{self.asset}_{self.window}_new_high_count'] = float(new_highs)

        # Recovery ratio: cumulative return / drawdown recovery time
        # rec_time = metrics.get('drawdown_recovery_time', 0.0)
        rec_time = recovery
        metrics[f'{self.asset}_{self.window}_recovery_ratio'] = float(cum_return / rec_time) if rec_time > 0 else 0.0

        # Skewness-to-volatility ratio
        # vol = metrics.get('realized_vol', 0.0)
        # skew = metrics.get('skewness', 0.0)
        metrics[f'{self.asset}_{self.window}_skew_vol_ratio'] = float(skewness / realized_vol) if realized_vol > 0 else 0.0

        n = r.size

        # Mean return and variance
        mean_r = float(np.mean(r)) if n > 0 else 0.0
        var_r = float(np.var(r)) if n > 0 and np.var(r) > 1e-8 else 0.0
        metrics[f'{self.asset}_{self.window}_mean_return'] = mean_r
        metrics[f'{self.asset}_{self.window}_variance'] = var_r

        # Range and IQR of returns
        if n > 0:
            metrics[f'{self.asset}_{self.window}_range_return'] = float(np.max(r) - np.min(r))
            metrics[f'{self.asset}_{self.window}_iqr_return'] = float(
                np.percentile(r, 75) - np.percentile(r, 25)
                )
        else:
            metrics[f'{self.asset}_{self.window}_range_return'] = 0.0
            metrics[f'{self.asset}_{self.window}_iqr_return'] = 0.0

        # Jarque-Bera statistic for normality
        S = skewness # metrics.get('skewness', 0.0)
        K = kurtosis # metrics.get('kurtosis', 0.0)
        metrics[f'{self.asset}_{self.window}_jarque_bera'] = float(
            n * (
                S**2 / 6.0 + K**2 / 24.0
                )
            ) if n > 0 else 0.0

        # Autocorrelation of squared returns (lag 1)
        if n > 1:
            r2 = r**2
            autocorr_sq = float(np.corrcoef(r2[:-1], r2[1:])[0, 1])
            if not np.isnan(autocorr_sq) and not np.isinf(autocorr_sq):
                metrics[f'{self.asset}_{self.window}_autocorr_sq'] = autocorr_sq # float(
                    # np.corrcoef(
                    #     r2[:-1], r2[1:]
                    #     )[0, 1]
                    # )
            else:
                metrics[f'{self.asset}_{self.window}_autocorr_sq'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_autocorr_sq'] = 0.0

        # Drawdown rate and entropy
        m = dd.size
        if m > 0:
            metrics[f'{self.asset}_{self.window}_drawdown_rate'] = float(np.sum(dd > 0) / m)
            # Entropy of drawdown distribution
            hist, _ = np.histogram(
                dd, bins=self.entropy_bins, density=True
                )
            probs = hist / np.sum(hist + 1e-8)
            probs = probs[probs > 0]
            metrics[f'{self.asset}_{self.window}_drawdown_entropy'] = float(
                -np.sum(
                    probs * np.log(probs)
                    )
                )
        else:
            metrics[f'{self.asset}_{self.window}_drawdown_rate'] = 0.0
            metrics[f'{self.asset}_{self.window}_drawdown_entropy'] = 0.0

        # Max drawdown length ratio
        # # max_dd_len = metrics.get('drawdown_length', 0.0)
        # max_dd_len = self._max_drawdown_length
        # metrics[f'{self.asset}_{self.window}_max_drawdown_length_ratio'] = (
        #     float(max_dd_len / self.window_size) if self.window_size > 0 else 0.0
        # )

        # Win/Loss ratio
        wins = np.sum(r > 0)
        losses = np.sum(r < 0)
        metrics[f'{self.asset}_{self.window}_win_loss_ratio'] = float(
            wins / losses) if losses != 0 else float(wins)

        # Average win streak and loss streak
        def avg_streak(arr, condition):
            streaks = []
            count = 0
            for val in arr:
                if condition(val):
                    count += 1
                else:
                    if count > 0:
                        streaks.append(count)
                        count = 0
            if count > 0:
                streaks.append(count)
            return float(np.mean(streaks)) if streaks else 0.0

        metrics[f'{self.asset}_{self.window}_avg_win_streak'] = avg_streak(r, lambda x: x > 0)
        metrics[f'{self.asset}_{self.window}_avg_loss_streak'] = avg_streak(r, lambda x: x < 0)

        # Tail return ratio: 95th percentile / abs(5th percentile)
        if r.size > 0:
            p95 = np.percentile(r, 95)
            p05 = np.percentile(r, 5)
            metrics[f'{self.asset}_{self.window}_tail_return_ratio'] = float(p95 / abs(p05)) if p05 != 0 else 0 # float('inf')
        else:
            metrics[f'{self.asset}_{self.window}_tail_return_ratio'] = 0.0

        # Spectral entropy of returns
        if r.size > 1:
            fft_vals = np.fft.fft(r)
            psd = np.abs(fft_vals)**2
            psd = psd[:len(psd)//2]  # keep positive freqs
            psd_sum = np.sum(psd)
            if psd_sum > 0:
                psd_norm = psd / psd_sum
                psd_norm = psd_norm[psd_norm > 0]
                metrics[f'{self.asset}_{self.window}_spectral_entropy'] = -float(
                    np.sum(
                        psd_norm * np.log(psd_norm)
                        )
                    )
            else:
                metrics[f'{self.asset}_{self.window}_spectral_entropy'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_spectral_entropy'] = 0.0

        # Average drawdown magnitude when in drawdown
        if dd.size > 0:
            dd_positive = dd[dd > 0]
            metrics[f'{self.asset}_{self.window}_avg_drawdown'] = float(
                np.mean(dd_positive)) if dd_positive.size > 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_avg_drawdown'] = 0.0


        # Annualized metrics (assume daily data, 252 trading days)
        metrics[f'{self.asset}_{self.window}_annual_return'] = float(
            np.mean(r) * 252) if n > 1 else 0.0
        
        metrics[f'{self.asset}_{self.window}_annual_volatility'] = float(
            np.std(r) * np.sqrt(252)) if n > 1 and np.std(r) > 1e-8 else 0.0
        
        if metrics[f'{self.asset}_{self.window}_annual_volatility'] > 0:
            metrics[f'{self.asset}_{self.window}_annual_sharpe'] = metrics[f'{self.asset}_{self.window}_annual_return'] / metrics[f'{self.asset}_{self.window}_annual_volatility']
        else:
            metrics[f'{self.asset}_{self.window}_annual_sharpe'] = 0.0

        # CAGR: compound annual growth rate
        """if len(self.asset_values) > 1:
            init = self.asset_values[0]
            curr = self.asset_values[-1]
            years = (
                len(self.asset_values) - 1
                ) / 252
            if init > 0 and years > 0:
                metrics[f'{self.asset}_{self.window}_cagr'] = float(
                    (curr / init) ** (1 / years) - 1
                    )
            else:
                metrics[f'{self.asset}_{self.window}_cagr'] = 0.0
        else:
            metrics[f'{self.asset}_{self.window}_cagr'] = 0.0"""

        # Trend slope & R-squared of portfolio values
        if pv.size > 1:
            x = np.arange(pv.size)
            slope, intercept = np.polyfit(x, pv, 1)
            yhat = slope * x + intercept
            ss_res = np.sum(
                (pv - yhat) ** 2
                )
            ss_tot = np.sum(
                (pv - np.mean(pv)) ** 2
                )
            metrics[f'{self.asset}_{self.window}_trend_slope'] = float(slope)
            metrics[f'{self.asset}_{self.window}_trend_r2'] = float(
                1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            metrics[f'{self.asset}_{self.window}_trend_slope'] = 0.0
            metrics[f'{self.asset}_{self.window}_trend_r2'] = 0.0

        # Return consistency: proportion of returns within ±1 std
        if n > 1:
            mean_r = np.mean(r)
            std_r = np.std(r) if np.std(r) > 1e-8 else 0.
            within = np.sum(np.abs(r - mean_r) <= std_r)
            metrics[f'{self.asset}_{self.window}_return_consistency'] = float(within / n)
        else:
            metrics[f'{self.asset}_{self.window}_return_consistency'] = 0.0

        # Zero-crossing rate of returns
        if n > 1:
            signs = np.sign(r)
            crossings = np.sum(signs[:-1] != signs[1:])
            metrics[f'{self.asset}_{self.window}_zero_crossing_rate'] = float(crossings / (n - 1))
        else:
            metrics[f'{self.asset}_{self.window}_zero_crossing_rate'] = 0.0

        metrics[
            f'{self.asset}_{self.window}_should_be_asset_value' # f'{self.asset}_{self.window}_should_be_asset_return'
            ] = float(
                self.should_be_asset_values[-1]) if len(
                    self.should_be_asset_values
                    ) > 0. else 0.
        metrics[
            f'{self.asset}_{self.window}_should_be_asset_return'
            ] = float(
                self.returns[-1]) if len(
                    self.returns
                    ) > 0. else 0.

        # hurst_exponent = self.hurst_exponent(r=r)
        # metrics[f'{self.asset}_{self.window}_hurst_exponent'] = hurst_exponent

        return metrics
        # if self.col_names is None:
        #     names = sorted(metrics.keys())
        # else:
        #     names = self.col_names
        # return np.array([metrics[name] for name in names], dtype=np.float32)




from collections import deque
from scipy.stats import skew, kurtosis

'''
How To Use In Your RL Env

On each environment step:
Call engine.record(self.portfolio_value, market_return=market_return, risk_free_rate=risk_free) after your agent’s action has updated the portfolio value.
To construct RL state:
Add engine.get_latest_features() into your state vector.
After an episode:
Use engine.as_dataframe() for analysis or reward computation.

'''

class PortfolioRiskFeatureEngine:
    """
    Computes rolling risk and alpha features on a dynamic portfolio.
    """
    def __init__(
        self,
        # window=252,
        window=30,
        maxlen=10000,
        risk_free_rate=0.000118, # 0.0,
        compute_market_beta=True,
        prefix="pf"
    ):
        self.window = window
        self.maxlen = maxlen
        self.prefix = prefix
        self.compute_market_beta = compute_market_beta

        self.portfolio_values = deque(maxlen=maxlen)
        self.market_returns = deque(maxlen=maxlen)
        self.risk_free_rates = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
        self.latest_features = {}
        self.static_risk_free = risk_free_rate

        self.portfolio_excess_returns = [] # deque(maxlen=maxlen) # []
        self.benchmark_excess_returns = [] # deque(maxlen=maxlen) # []

        self.col_names = [f"{self.prefix}_alpha_monthly",
            f"{self.prefix}_vol_monthly",
            f"{self.prefix}_skew",
            f"{self.prefix}_kurtosis",
            f"{self.prefix}_cvar_05",
            f"{self.prefix}_sharpe",
            f"{self.prefix}_sortino",
            f"{self.prefix}_max_drawdown",
            f"{self.prefix}_m_squared_ratio",
            f"{self.prefix}_beta",
            f"{self.prefix}_beta_adj_sharpe",]
        self.latest_features.update({col: 0 for col in self.col_names})
          

    def reset(self):
        self.portfolio_values.clear()
        self.market_returns.clear()
        self.risk_free_rates.clear()
        self.timestamps.clear()
        # self.latest_features = {}

        self.portfolio_excess_returns.clear()
        self.benchmark_excess_returns.clear()

        self.latest_features.update(
            {
                col: 0 for col in self.col_names
                }
            )



    def record(self, portfolio_value, market_return=0.0, risk_free_rate=None, timestamp=None):
        self.portfolio_values.append(portfolio_value)
        self.market_returns.append(market_return)

        # self.portfolio_excess_returns.append(
        if risk_free_rate is None:
            if isinstance(self.static_risk_free, (float, int)):
                self.risk_free_rates.append(self.static_risk_free)
            else:
                self.risk_free_rates.append(
                    self.static_risk_free[len(self.portfolio_values) - 1]
                )
        else:
            self.risk_free_rates.append(risk_free_rate)
        self.timestamps.append(timestamp)

        risk_free = self.risk_free_rates[-1]
        values = np.array(self.portfolio_values)[:]

        # print(f'values: {values}')
        # port_rets = np.diff(self.portfolio_values) / self.portfolio_values[:-1] # if len(self.portfolio_values) > 1 else self.portfolio_values[-1]
        port_rets = np.diff(values) / values[:-1] # if values.size > 1 else values[-1]
        # port_rets = np.insert(port_rets, 0, 0.0)
        excess_port = port_rets - risk_free
        excess_market = np.mean(self.market_returns) - risk_free

        # print(f'port_rets: {port_rets} ||| excess_port: {excess_port} ||| excess_market: {excess_market}')
        # values = np.array(self.portfolio_values)
        # market_ret = np.array(self.market_returns)
        # risk_free = np.array(self.risk_free_rates)

        # port_rets = np.diff(self.portfolio_values) / self.portfolio_values[:-1] if len(self.portfolio_values) > 1 else self.portfolio_values[-1]
        # port_rets = np.insert(port_rets, 0, 0.0)
        # excess_port = port_rets - risk_free
        # excess_market = market_ret - risk_free
        # if values.size < 1:
        # self.portfolio_excess_returns.append(excess_port)
        self.benchmark_excess_returns.append(excess_market)
        # else:
        self.portfolio_excess_returns.extend(excess_port)
        # self.benchmark_excess_returns.extend(excess_market)

        # self.latest_features = self._compute_features()
        # m_squared_ratio = self.m_squared_ratio(risk_free_rate=risk_free[-1])

        if len(self.portfolio_values) >= self.window:
            self.latest_features = self._compute_features()
        else:
            self.latest_features.update({col: 0 for col in self.col_names})
        #     self.latest_features = {}

    # Function to calculate M-squared ratio
    # def m_squared_ratio(window_data):
    def m_squared_ratio(self, risk_free_rate, debug=False):
        # portfolio_excess_returns = self.portfolio_excess_returns # excess_port # window_data['portfolio_excess_return']
        # benchmark_excess_returns = self.benchmark_excess_returns    # excess_market # window_data['benchmark_excess_return']
        # risk_free_rate = window_data[risk_free_rate_col].iloc[0]
        # print(f'self.portfolio_excess_returns: {self.portfolio_excess_returns} ||| self.benchmark_excess_returns: {self.benchmark_excess_returns}')
        if len(self.portfolio_values) >= 3:
            
            # values = np.array(self.portfolio_excess_returns)
            # b_values = np.array(self.benchmark_excess_returns)
            portfolio_std = np.std(self.portfolio_excess_returns)
            benchmark_std = np.std(self.benchmark_excess_returns)

            if portfolio_std == 0 or benchmark_std == 0:
                return 0.

            elif portfolio_std > 1e-8 and benchmark_std > 1e-8:
                portfolio_mean = np.mean(self.portfolio_excess_returns)
                adjusted_portfolio_performance = risk_free_rate + (
                    (portfolio_mean - risk_free_rate) * (benchmark_std / portfolio_std)
                    )
                # return ( # Original.
                m_squared_ratio = float(
                    (adjusted_portfolio_performance - risk_free_rate) - (
                        np.mean(self.benchmark_excess_returns) - risk_free_rate
                        )
                    ) * (benchmark_std / portfolio_std)
                if debug:
                    print(
f'''
||| portfolio_mean: {portfolio_mean}
||| adjusted_portfolio_performance: {adjusted_portfolio_performance}
||| benchmark_std: {benchmark_std}
||| portfolio_std: {portfolio_std}
||| risk_free_rate: {risk_free_rate}
||| m_squared_ratio: {m_squared_ratio}
'''
                    )
                return m_squared_ratio
            else:
                return 0.0
        else:
            return 0.0


    def _compute_features(self, debug=False):
        values = np.array(self.portfolio_values)[-self.window:]
        market_ret = np.array(self.market_returns)[-self.window:]
        risk_free = np.array(self.risk_free_rates)[-self.window:]

        port_rets = np.diff(values) / values[:-1]
        port_rets = np.insert(port_rets, 0, 0.0) # 0 for first

        # Excess returns
        excess_port = port_rets - risk_free
        excess_market = market_ret - risk_free

        # self.portfolio_excess_returns.extend(excess_port) # Original.
        # self.benchmark_excess_returns.extend(excess_market) # Original.
        # self.portfolio_excess_returns.append(excess_port)
        # self.benchmark_excess_returns.append(excess_market)
        # print(f'||| self.portfolio_excess_returns: {self.portfolio_excess_returns} ||| excess_port: {excess_port} ||| risk_free[-1]: {risk_free[-1]} ||| self.benchmark_excess_returns: {self.benchmark_excess_returns} ||| excess_market: {excess_market}')
        # stop
        # m_squared_ratio = self.m_squared_ratio(risk_free_rate=risk_free[-1], debug=debug)

        # Downside
        mean_excess = np.mean(excess_port)
        # std_excess = np.std(excess_port) + 1e-12
        # downside = np.std(np.minimum(0, excess_port)) + 1e-12
        if np.std(excess_port) > 1e-8:
            std_excess = np.std(excess_port) + 1e-12
        else:
            std_excess = 0.0
        if np.std(np.minimum(0, excess_port)) > 1e-8:
            downside = np.std(np.minimum(0, excess_port)) + 1e-12
        else:
            downside = 0.0

        # Alpha (annualized mean excess return)
        alpha_annual = mean_excess * 252 # 7

        # Volatility (annualized)
        vol_annual = np.std(port_rets) * np.sqrt(252) # 7

        # Skewness and Kurtosis
        skewness = skew(port_rets)
        kurt = kurtosis(port_rets)

        # CVaR (Conditional Value-at-Risk, 5%)
        q05 = np.percentile(port_rets, 5)
        cvar_05 = port_rets[port_rets <= q05].mean() if np.any(port_rets <= q05) else 0.0

        # Rolling Sharpe
        # sharpe = mean_excess / std_excess
        if std_excess > 1e-8:
            sharpe = mean_excess / std_excess
        else:
            sharpe = 0.0
        if downside > 1e-8:
            sortino = mean_excess / downside
        else:
            sortino = 0.0
        # Sortino
        # sortino = mean_excess / downside

        # Max Drawdown
        cumulative = np.cumprod(1 + port_rets)
        drawdowns = 1 - cumulative / np.maximum.accumulate(cumulative)
        max_drawdown = np.max(drawdowns)

        m_squared_ratio = self.m_squared_ratio(risk_free_rate=risk_free[-1], debug=debug)

        features = {
            # f"{self.prefix}_alpha_annual": alpha_annual,
            # f"{self.prefix}_vol_annual": vol_annual,
            f"{self.prefix}_alpha_monthly": alpha_annual,
            f"{self.prefix}_vol_monthly": vol_annual,
            # f"{self.prefix}_skew": skewness,
            # f"{self.prefix}_kurtosis": kurt,
            # f"{self.prefix}_cvar_05": cvar_05,
            f"{self.prefix}_sharpe": sharpe,
            f"{self.prefix}_sortino": sortino,
            f"{self.prefix}_max_drawdown": max_drawdown,
            f"{self.prefix}_m_squared_ratio": m_squared_ratio,
        }
        # print(f'm_squared_ratio: {m_squared_ratio}')

        if len(self.portfolio_values) >= self.window:
            if self.compute_market_beta:
                if np.std(excess_market) > 1e-8:
                    beta = np.cov(excess_port, excess_market)[0, 1] / np.var(excess_market)
                else:
                    beta = 0.0
                if beta > 1e-8:
                    basr = sharpe / (beta)
                else:
                    basr = 0.0

                features[f"{self.prefix}_beta"] = beta
                features[f"{self.prefix}_beta_adj_sharpe"] = basr
                if debug:
                    print(
f'''
||| np.std(excess_market): {np.std(excess_market)}
||| excess_port.mean(): {np.mean(excess_port)}
||| excess_market.std(): {np.std(excess_market)}
||| np.std(excess_market): {np.std(excess_market)}
||| np.cov(excess_port, excess_market)[0, 1]: {np.cov(excess_port, excess_market)[0, 1]}
||| np.var(excess_market): {np.var(excess_market)}
||| beta_adj_sharpe: {basr}
'''
                    )
        else:
            features[f"{self.prefix}_beta"] = 0
            features[f"{self.prefix}_beta_adj_sharpe"] = 0
        return features

    def get_latest_features(self):
        return self.latest_features.copy()

    def as_dataframe(self):
        feats = []
        vals = np.array(self.portfolio_values)
        market = np.array(self.market_returns)
        risk_free = np.array(self.risk_free_rates)
        ts = np.array(self.timestamps)
        for i in range(self.window, len(vals) + 1):
            v = vals[i - self.window:i]
            m = market[i - self.window:i]
            r = risk_free[i - self.window:i]
            t = ts[i - 1]  # last timestamp in window

            port_rets = np.diff(v) / v[:-1]
            port_rets = np.insert(port_rets, 0, 0.0)
            excess_port = port_rets - r
            excess_market = m - r

            mean_excess = np.mean(excess_port)
            std_excess = np.std(excess_port) # + 1e-12
            downside = np.std(np.minimum(0, excess_port)) # + 1e-12
            alpha_annual = mean_excess * 252 # 7 # 252
            vol_annual = np.std(port_rets) * np.sqrt(252) # 7) # 252)
            skewness = skew(port_rets)
            kurt = kurtosis(port_rets)
            q05 = np.percentile(port_rets, 5)
            cvar_05 = port_rets[port_rets <= q05].mean() if np.any(port_rets <= q05) else 0.0
            if std_excess > 1e-12:
                sharpe = mean_excess / std_excess
            else:
                sharpe = 0.
            if downside > 1e-12:
                sortino = mean_excess / downside
            else:
                sortino = 0.
            cumulative = np.cumprod(1 + port_rets)
            drawdowns = 1 - cumulative / np.maximum.accumulate(cumulative)
            max_drawdown = np.max(drawdowns)

            f = {
                "timestamp": t,
                f"{self.prefix}_alpha_annual": alpha_annual,
                f"{self.prefix}_vol_annual": vol_annual,
                f"{self.prefix}_skew": skewness,
                f"{self.prefix}_kurtosis": kurt,
                f"{self.prefix}_cvar_05": cvar_05,
                f"{self.prefix}_sharpe": sharpe,
                f"{self.prefix}_sortino": sortino,
                f"{self.prefix}_max_drawdown": max_drawdown,
            }
            if self.compute_market_beta:
                if np.std(excess_market) > 1e-8:
                    beta = np.cov(excess_port, excess_market)[0, 1] / np.var(excess_market)
                else:
                    beta = 0.0
                basr = sharpe / (beta + 1e-12)
                f[f"{self.prefix}_beta"] = beta
                f[f"{self.prefix}_beta_adj_sharpe"] = basr
            feats.append(f)
        return pd.DataFrame(feats)

from collections import deque








# import numpy as np
# import warnings
# import logging

from typing import Union
import math

# logger = logging.getLogger(__name__)

# import numpy as np
# import warnings
# import logging
from typing import Any

# logger = logging.getLogger(__name__)

def safe_corrcoef(
    X: np.ndarray, 
    *,
    ddof: int = 1
) -> np.ndarray:
    """
    Compute a correlation matrix for the columns of X, handling zero‐std columns
    and suppressing invalid‐value warnings.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix.
    ddof : int, default=1
        Delta degrees of freedom for covariance (ddof=1 for sample covariance).

    Returns
    -------
    corr : np.ndarray, shape (n_features, n_features)
        The correlation matrix, with any NaNs/Infs replaced by 0.
    """
    X = np.asarray(X, dtype=float)
    # Center the data
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # Suppress warnings for the next block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Compute covariance matrix
        cov = (Xc.T @ Xc) / (X.shape[0] - ddof)
        # Extract standard deviations
        stddev = np.sqrt(np.diag(cov))

        # Avoid dividing by zero
        zero_mask = stddev == 0
        if np.any(zero_mask):
            logger.warning(
                f"safe_corrcoef: found {zero_mask.sum()} constant feature(s) with zero std; "
                "their correlations will be set to zero."
            )
            stddev[zero_mask] = 1.0  # so we don't divide by zero

        # Build correlation matrix
        corr = cov / stddev[:, None] / stddev[None, :]

    # Replace any NaN/Inf with zero
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr





def safe_cov(
    X: Union[np.ndarray, list],
    Y: Union[np.ndarray, list, None] = None,
    *,
    ddof: int = 1,
    rowvar: bool = False
) -> Union[float, np.ndarray]:
    """
    A “safe” covariance estimator.

    - If `Y` is provided, returns the scalar covariance between X and Y.
    - If `Y` is None:
        • 1D X → returns scalar variance of X.
        • 2D X → returns covariance matrix, treating columns as variables
          (or rows if rowvar=True).

    Any divide-by-zero or invalid results become 0.0, and a warning
    is logged if n_obs ≤ ddof.
    """
    X = np.asarray(X, dtype=float)

    # --- Two-series case: return scalar covariance(X, Y) ---
    if Y is not None:
        Y = np.asarray(Y, dtype=float)
        if X.shape != Y.shape:
            raise ValueError(f"safe_cov: X and Y must have same shape, got {X.shape} vs {Y.shape}")

        n_obs = X.size
        if n_obs - ddof <= 0:
            logger.warning(f"safe_cov: only {n_obs} observations for ddof={ddof}, covariance set to 0.0")

        Xc = X - X.mean()
        Yc = Y - Y.mean()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cov_xy = (Xc @ Yc) / (n_obs - ddof)

        return float(np.nan_to_num(cov_xy, nan=0.0, posinf=0.0, neginf=0.0))

    # --- Single-series case: 1D variance ---
    if X.ndim == 1:
        n_obs = X.size
        if n_obs - ddof <= 0:
            logger.warning(f"safe_cov: only {n_obs} observations for ddof={ddof}, variance set to 0.0")

        Xc = X - X.mean()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            var = (Xc @ Xc) / (n_obs - ddof)

        return float(np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0))

    # --- Full matrix case: 2D array →
    if X.ndim == 2:
        if rowvar:
            X = X.T  # samples must be rows
        n_obs, n_vars = X.shape
        if n_obs - ddof <= 0:
            logger.warning(f"safe_cov: only {n_obs} observations for ddof={ddof}, covariance matrix may be zeroed")

        mean = X.mean(axis=0, keepdims=True)
        Xc = X - mean

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cov = (Xc.T @ Xc) / (n_obs - ddof)

        return np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    raise ValueError(f"safe_cov: unsupported array shape {X.shape}")



# =============================================================================
# LIVE PnL CALCULATOR (FIFO and LIFO) WITH INDIVIDUAL ASSET UPDATES
# =============================================================================
class LivePnLCalculator:
    def __init__(self, assets: List[str]):
        self.assets = assets
        # Maintain separate books for FIFO (as a deque) and LIFO (as a list)
        self.fifo_positions = {asset: deque() for asset in assets}
        self.lifo_positions = {asset: [] for asset in assets}
        self.realized_pnl_fifo = {asset: 0.0 for asset in assets}
        self.realized_pnl_lifo = {asset: 0.0 for asset in assets}
        # Track the last observed price for each asset (required for hold PnL)
        # self.last_price = {asset: None for asset in assets}
        self.last_price = {asset: 0 for asset in assets}
        logger.info("LivePnLCalculator initialized for assets: %s", assets)

    def reset(self):
        for asset in self.assets:
            # Maintain separate books for FIFO (as a deque) and LIFO (as a list)
            self.fifo_positions[asset].clear()
            self.lifo_positions[asset].clear()
            self.realized_pnl_fifo[asset] = 0.0
            self.realized_pnl_lifo[asset] = 0.0
            # Track the last observed price for each asset (required for hold PnL)
            # self.last_price = {asset: None for asset in assets}
            self.last_price[asset] = 0

        logger.info("LivePnLCalculator reseted for assets: %s", self.assets)

    def process_trade(self, asset: str, volume: float, price: float, trade_type: str) -> None:
        """
        Process a trade and update both FIFO and LIFO positions as well as realized PnL.
        Parameters:
          - asset (str): The asset symbol.
          - volume (float): Trade volume (positive for buy, negative for sell).
          - price (float): Trade price.
          - trade_type (str): "buy" or "sell".
        """
        if asset not in self.fifo_positions or asset not in self.lifo_positions:
            logger.error("Asset %s not found in position books.", asset)
            raise ValueError(f"Asset {asset} is not managed by this PnL calculator.")
        # Always update the last seen price, even on 'hold'
        self.last_price[asset] = price
        try:
            if trade_type == "buy":
                position = {'volume': volume, 'price': price}
                self.fifo_positions[asset].append(position)
                self.lifo_positions[asset].append(position)
                logger.debug("Processed BUY for %s: volume=%.4f, price=%.4f", asset, volume, price)
            elif trade_type == "sell":
                sell_volume = abs(volume)
                # Process FIFO:
                while sell_volume > 0 and self.fifo_positions[asset]:
                    pos = self.fifo_positions[asset][0]
                    if pos['volume'] <= sell_volume:
                        realized = pos['volume'] * (price - pos['price'])
                        self.realized_pnl_fifo[asset] += realized
                        sell_volume -= pos['volume']
                        self.fifo_positions[asset].popleft()
                    else:
                        realized = sell_volume * (price - pos['price'])
                        self.realized_pnl_fifo[asset] += realized
                        pos['volume'] -= sell_volume
                        sell_volume = 0
                # Process LIFO:
                sell_volume = abs(volume)
                while sell_volume > 0 and self.lifo_positions[asset]:
                    pos = self.lifo_positions[asset][-1]
                    if pos['volume'] <= sell_volume:
                        realized = pos['volume'] * (price - pos['price'])
                        self.realized_pnl_lifo[asset] += realized
                        sell_volume -= pos['volume']
                        self.lifo_positions[asset].pop()
                    else:
                        realized = sell_volume * (price - pos['price'])
                        self.realized_pnl_lifo[asset] += realized
                        pos['volume'] -= sell_volume
                        sell_volume = 0
                logger.debug("Processed SELL for %s: volume=%.4f, price=%.4f", asset, volume, price)
            elif trade_type == "hold":
                # This will always compute correct realized/unrealized based on last book state.
                # fifo_r, fifo_u, lifo_r, lifo_u = self.update_asset_pnl(asset, price, timestamp)
                # logger.debug("Processed HOLD for %s at %s: FIFO_realized=%.4f, FIFO_unrealized=%.4f, LIFO_realized=%.4f, LIFO_unrealized=%.4f",
                #     asset, timestamp, fifo_r, fifo_u, lifo_r, lifo_u)

                # For "hold", you only need to ensure last_price is updated
                # No change in position or realized PnL
                logger.info("Processed HOLD for %s: price=%.4f", asset, price)
            else:
                raise ValueError("trade_type must be 'buy' or 'sell'")
        except Exception as e:
            logger.error("Error processing trade for %s: %s", asset, str(e))
            raise

    def update_asset_pnl(self, asset: str, current_price: float, timestamp: pd.Timestamp) -> Tuple[float, float, float, float]:
        """
        Compute and log the current realized and unrealized PnL for a given asset.
        Returns a tuple of (FIFO_realized, FIFO_unrealized, LIFO_realized, LIFO_unrealized).
        """
        if current_price is None:
            current_price = self.last_price.get(asset, 0.0) or 0.0
        try:
            fifo_realized = float(self.realized_pnl_fifo.get(asset, 0.0))
            fifo_unrealized = sum(pos['volume'] * (current_price - pos['price']) for pos in self.fifo_positions[asset])
            if not self.fifo_positions[asset]:
                fifo_unrealized = 0.0
            lifo_realized = float(self.realized_pnl_lifo.get(asset, 0.0))
            lifo_unrealized = sum(pos['volume'] * (current_price - pos['price']) for pos in self.lifo_positions[asset])
            if not self.lifo_positions[asset]:
                lifo_unrealized = 0.0
            logger.debug("Updated PnL for %s at %s: FIFO_realized=%.4f, FIFO_unrealized=%.4f, "
                         "LIFO_realized=%.4f, LIFO_unrealized=%.4f",
                         asset, timestamp, fifo_realized, fifo_unrealized, lifo_realized, lifo_unrealized)
            return (
                fifo_realized,
                fifo_unrealized,
                lifo_realized,
                lifo_unrealized,
            )
        except Exception as e:
            logger.error("Error updating PnL for %s: %s", asset, str(e))
            raise


    # Getters: always use the most recent price seen (or allow override)
    def get_realized_pnl_fifo(self, asset: str) -> float:
        return float(self.realized_pnl_fifo.get(asset, 0.0))

    def get_unrealized_pnl_fifo(self, asset: str, current_price: float = None) -> float:
        if current_price is None:
            current_price = self.last_price.get(asset, 0.0) or 0.0
        return float(sum(pos['volume'] * (current_price - pos['price']) for pos in self.fifo_positions[asset])) if self.fifo_positions[asset] else 0.0

    def get_realized_pnl_lifo(self, asset: str) -> float:
        return float(self.realized_pnl_lifo.get(asset, 0.0))

    def get_unrealized_pnl_lifo(self, asset: str, current_price: float = None) -> float:
        if current_price is None:
            current_price = self.last_price.get(asset, 0.0) or 0.0
        return float(sum(pos['volume'] * (current_price - pos['price']) for pos in self.lifo_positions[asset])) if self.lifo_positions[asset] else 0.0

    def get_total_pnl(self, asset: str, current_price: float, method: str = "FIFO") -> float:
        if method.upper() == "FIFO":
            return self.get_realized_pnl_fifo(asset) + self.get_unrealized_pnl_fifo(asset, current_price)
        elif method.upper() == "LIFO":
            return self.get_realized_pnl_lifo(asset) + self.get_unrealized_pnl_lifo(asset, current_price)
        else:
            raise ValueError("Method must be either 'FIFO' or 'LIFO'")

    def get_all_asset_pnls(self) -> Dict[str, Dict[str, float]]:
        """Convenience: Return a full dict of PnLs for all assets, using last seen price."""
        out = {}
        for asset in self.fifo_positions:
            last = self.last_price.get(asset, 0.0) or 0.0
            out[asset] = {
                f'{asset}_fifo_realized': self.get_realized_pnl_fifo(asset),
                f'{asset}_fifo_unrealized': self.get_unrealized_pnl_fifo(asset, last),
                f'{asset}_lifo_realized': self.get_realized_pnl_lifo(asset),
                f'{asset}_lifo_unrealized': self.get_unrealized_pnl_lifo(asset, last),
            }
        return out


def calculate_cvar(
    data,
    confidence_level=0.95,
    method="historical",
    input_is_returns=False
):
    """
    Calculate Conditional Value at Risk (CVaR, or Expected Shortfall) for a time series.
    
    Args:
        data (list, np.array): Portfolio values or returns.
        confidence_level (float): CVaR confidence level (e.g., 0.95 for 95% CVaR).
        method (str): "historical" or "parametric".
        input_is_returns (bool): If True, 'data' is already returns; else, compute from values.

    Returns:
        float: CVaR (as a negative float, e.g., -0.03 for -3% loss), or 0.0 if not enough data.
    """
    arr = np.asarray(data)
    # Ensure returns
    if not input_is_returns:
        if len(arr) < 2:
            return 0.0
        returns = arr[1:] / arr[:-1] - 1
    else:
        returns = arr

    # Not enough data for a robust estimate
    if len(returns) < 2:
        return 0.0

    if method.lower() == "historical":
        # Calculate the VaR threshold
        percentile = 100 * (1 - confidence_level)
        var_threshold = np.percentile(returns, percentile)
        # Losses worse than or equal to VaR
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) == 0:
            return 0.0
        cvar = np.mean(tail_losses)
    elif method.lower() == "parametric":
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        if np.isnan(sigma) or sigma == 0 or np.isnan(mu):
            return 0.0
        alpha = 1 - confidence_level
        z = norm.ppf(alpha)
        # Analytical formula for normal distribution
        cvar = mu - sigma * norm.pdf(z) / alpha
    else:
        raise ValueError("method must be 'historical' or 'parametric'")

    if np.isnan(cvar):
        return 0.0
    return float(cvar)




class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,
                 data=None,
                 selected_assets=None,
                 initial_balance=1000,
                 transaction_fee=0.001,
                 start_step=3500,
                 scaler=StandardScaler, # MinMaxScaler,
                 running_scaler=StandardScaler, # MinMaxScaler,
                 train=True,
                 # historical_trades=None
                 price_data=None,
                 seq_len=30,
                 train_ratio=0.8,
                 buffer_size=1000,
                 env_name='CryptoTradingEnv',
                 use_2d=False,
                 if_discrete=False,
                 use_seq_obs=False,
                 use_action_norm=False,
                 # window_size=7,
                 fixed_fee=1.0, # Flat fee per trade (e.g., $1)
                 min_trade_value=10.0, # Minimum notional trade value to execute
                 risk_free_rate=0.000118, #   -> Daily risk free rate. # (0.0463) -> yearly risk free rate.
                 ema_short_period: int = 12,
                 ema_long_period: int = 26,
                 rsi_window: int = 14,
                 entropy_bins: int = 10,
                 var_ratio_lag: int = 5,
                 horizon: Optional[int] = None,
                 reward_weights=dict(
                     pf_sharpe=1.0,
                     log_return=1.0,
                     vol_monthly=0.05,
                     pf_max_drawdown=0.05,
                     alpha_monthly=1.0,
                     m_squared_ratio=0.05,
                     beta_adj_sharpe=0.5,
                     pf_cvar=0.001,
                     pf_sortino=0.1,
                     pf_beta=0.1,
                     blocked_actions_w=0.001,
                     tx_fee_w=0.002,
                     prospect_theory_loss_aversion_alpha=2.0,
                     prospect_theory_loss_aversion_w=0.5,
                     ),
                 ):
        self.horizon = horizon
        # Buffers for new metrics
        self.realized_pnls: List[float] = []
        self.cumulative_returns: List[float] = []

        # Buffers for new features
        self.cash_ratios: List[float] = []
        self.asset_counts: List[float] = []
        self.concentration_ratios: List[float] = []
        self.trade_counts: List[float] = []
        self.value_zscores: List[float] = []

        self.entropy_bins = entropy_bins
        self.var_ratio_lag = var_ratio_lag
        self.momentum_window = seq_len # momentum_window or seq_len
        self.reward_weights = reward_weights
        self.use_action_norm = use_action_norm
        end_step = len(data)
        max_ep_len = int(end_step - start_step)

        self.fixed_fee = fixed_fee # 1.0             # Flat fee per trade (e.g., $1)
        self.proportional_fee = transaction_fee # 0.002    # Proportional fee (e.g., 0.1%)
        self.min_trade_value = min_trade_value  # 10.0      # Minimum notional trade value to execute
        self.engine = PortfolioRiskFeatureEngine(
            window=seq_len,
            maxlen=int(max_ep_len),
            risk_free_rate=risk_free_rate # 0.000118 #  -> Daily risk free rate. # 0.0463) -> yearly risk free rate.
            )

        self.risk_free_rate_ = risk_free_rate
        self.returns = []

        # EMA parameters
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.alpha_short = 2 / (ema_short_period + 1)
        self.alpha_long = 2 / (ema_long_period + 1)
        self.prev_ema_short: Optional[float] = None
        self.prev_ema_long: Optional[float] = None

        # RSI window
        self.rsi_window = rsi_window

        self.seq_len = seq_len
        self.use_seq_obs=use_seq_obs
        self.initialized = False
        self.if_discrete = if_discrete
        self.use_2d = use_2d
        self.env_name = env_name
        self.mean = None
        self.std = None
        self.should_be_portfolio_value = float(0.)
        self.should_be_portfolio_values = []

        self.portfolio_metrics_col = [
            'sharpe',
            'sortino',
            'momentum',
            'skewness',
            'kurtosis',
            'realized_vol',
            'ulcer_index',
            'vol_of_vol',
            'beta',
            'correlation',
            # 'omega',
            'calmar_mean_r',
            'var',
            'cvar',
            'max_drawdown',
            'max_drawdown_length',
            # 'turnover',
            # 'transaction_cost',
            'entropy',
            'risk_parity_deviation',
            'excess_return',
            'tracking_error',
            'information_ratio',
            # 'ema_short',
            # 'ema_long',
            'macd',
            # 'rsi',
            # 'win_rate',
            # 'profit_factor',
            # 'rachev',
            'calmar_cum_r',
            'drawdown_recovery_time',
            'autocorr_lag1',
            'hurst',
            'efficiency_ratio',
            'kelly_fraction',
            'last_return_z',
            'price_zscore',
            'mfe',
            'mae',
            # 'tail_variance',
            'downside_vol',
            'momentum_zscore',
            'return_entropy',
            'variance_ratio',
            # 'win_streak',
            # 'loss_streak',
            'effective_bets',
            'mad',
            'gini',
            # 'hill_estimator',
            'drawdown_at_risk',
            'cov',
            'cash_ratio',
            'asset_count',
            'concentration_ratio',
            'trade_count',
            'portfolio_value_zscore',
            'avg_win_return',
            'avg_loss_return',
            'max_single_drawdown',
            'avg_trade_size',
            'realized_pnl',
            'cum_return',
            # 'episode_progress',
            # 'sterling_ratio',
            'ulcer_ratio',
            'cdar',
            # 'avg_holding_period',
            # 'new_high_count',
            'recovery_ratio',
            'skew_vol_ratio',
            'mean_return',
            'variance',
            'range_return',
            'iqr_return',
            'jarque_bera',
            'autocorr_sq',
            'drawdown_rate',
            'drawdown_entropy',
            # 'win_loss_ratio',
            # 'avg_win_streak',
            # 'avg_loss_streak',
            # 'tail_return_ratio',
            'spectral_entropy',
            'avg_drawdown',
            'annual_return',
            'annual_volatility',
            'annual_sharpe',
            'cagr',
            'trend_slope',
            'trend_r2',
            'return_consistency',
            # 'zero_crossing_rate'
            ]

        self.metrics = {}

        self.current_weights = [] # self.get_current_weights # 27483
        self.previous_weights = [] # self.current_weights # [] # self.get_current_weights # [1:] # [] # self.current_weights
        self.hrp_previous_weights = []

        self.buffer_size = buffer_size
        self.running_scaler = MinMaxScaler(feature_range=(0, 1)) # RobustScaler() # StandardScaler() # MinMaxScaler(feature_range=(0, 1))
  
        self.scaler = deepcopy(scaler)
 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.train = train
        self.seq_len = seq_len # Original.

        if train:
            self.start_step = start_step
        else:
            self.start_step = start_step # seq_len + 1
        self.start_step_train = start_step
            
        self.train_ratio = train_ratio

        # selected_assets = ['Close', 'ETH Close', 'SP500 Close']
        if data is not None and selected_assets is not None:
            columns = ['timestamp']
            columns += selected_assets
            # data, selected_asset, price_data = self.init_data()
        else:
            data, selected_assets, price_data = self.init_data()


        data_copy =  deepcopy(data)
        return_data = deepcopy(data)

        asset_prices_df = price_data[selected_assets].copy() # pd.DataFrame(price_data, columns=price_data.drop(price_data['timestamp'], inplace=False))
        # print(f'asset_prices_df: {asset_prices_df}')
        self.returns_df = deepcopy(asset_prices_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(value=0))
 
        if self.use_2d:
            self.data = deepcopy(price_data)
        else:
            data = data.set_index('timestamp', inplace=False, drop=True)
            self.data = deepcopy(pd.DataFrame(self.scaler.transform(data), index=data.index, columns=data.columns))
        self.data.reset_index(inplace=True, drop=False)
        print(f'||| self.data: {self.data} ||| len(data) : {len(data)} ||| start_step: {start_step} ||| end_step: {end_step}')
        self.data.set_index('timestamp', inplace=True)
 
        self.price_data = price_data


        init_volume = 1000 / self.price_data.iloc[self.start_step]['ETH Close'] # Original.
        btc_init_volume = 1000 / self.price_data.iloc[self.start_step]['Close'] # Original.
        self.current_step = self.start_step

        # Ensure 'timestamp' is a column in self.data
        if 'timestamp' not in self.data.columns:
            self.data.reset_index(inplace=True)
            self.data.rename(columns={'index': 'timestamp'}, inplace=True)
            # self.data.rename(columns={'index': 'timestamp'}, inplace=True)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        if train:
            self.end_step = len(data) # - 1 # - 1
            # self.end_step = start_step + int((len(data) - start_step) * 0.8)
            self.flag = 'Training'
        else:
            self.end_step = len(data) # - 1 # - 1
            self.flag = 'Testing'

        self.steps_per_epoch = self.end_step - self.start_step - 1
        print(f'Currently under {self.flag}, the start step is {self.start_step}, the end step is {self.end_step}')

        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        # self.balance = initial_balance
        self.index = len(selected_assets)
        # self.cash_reserve = initial_balance

        self.selected_assets = selected_assets # List of asset names to be included
        self.portfolio = {'cash': initial_balance}
        self.portfolio.update({asset: 0 for asset in selected_assets})


        self.windw_sizes = [30]
        self.fe_dict = {
            f"{asset}_{ws}_Features_Engine": FeatureEngine(
                window_size=ws,
                maxlen=max_ep_len,
                risk_free_rate=0.000118,
                compute_market_beta=True,
                asset=asset,
                prefix=asset,
            )
            for asset, ws in product(self.selected_assets, self.windw_sizes)
            # for asset, ws in product(self.portfolio.keys(), self.windw_sizes)
        }

        for asset, window_size in product(self.selected_assets, self.windw_sizes):
            if asset != 'cash':
                self.portfolio_metrics_col += [f'{asset}_{window_size}_cash_ratio'] # ,  f'{asset}_{window_size}_effective_bets']
            self.portfolio_metrics_col += [
                f'{asset}_{window_size}_should_be_asset_value',
                f'{asset}_{window_size}_should_be_asset_return',
                f'{asset}_{window_size}_sharpe',
                f'{asset}_{window_size}_sortino',
                f'{asset}_{window_size}_momentum',
                f'{asset}_{window_size}_skewness',
                f'{asset}_{window_size}_kurtosis',
                f'{asset}_{window_size}_realized_vol',
                f'{asset}_{window_size}_ulcer_index',
                f'{asset}_{window_size}_vol_of_vol',
                f'{asset}_{window_size}_beta',
                f'{asset}_{window_size}_correlation',
                f'{asset}_{window_size}_omega',
                f'{asset}_{window_size}_calmar_mean_r',
                f'{asset}_{window_size}_var',
                f'{asset}_{window_size}_cvar',
                f'{asset}_{window_size}_max_drawdown',
                f'{asset}_{window_size}_max_drawdown_length',
                # 'turnover',
                # 'transaction_cost',
                # f'{asset}_{window_size}_entropy',
                # f'{asset}_{window_size}_risk_parity_deviation',
                f'{asset}_{window_size}_excess_return',
                f'{asset}_{window_size}_tracking_error',
                f'{asset}_{window_size}_information_ratio',
                f'{asset}_{window_size}_ema_short',
                f'{asset}_{window_size}_ema_long',
                # f'{asset}_{window_size}_macd',
                f'{asset}_{window_size}_rsi',
                # f'{asset}_{window_size}_win_rate',
                f'{asset}_{window_size}_profit_factor',
                f'{asset}_{window_size}_rachev',
                f'{asset}_{window_size}_calmar_cum_r',
                f'{asset}_{window_size}_drawdown_recovery_time',
                f'{asset}_{window_size}_autocorr_lag1',
                f'{asset}_{window_size}_hurst',
                f'{asset}_{window_size}_efficiency_ratio',
                f'{asset}_{window_size}_kelly_fraction',
                f'{asset}_{window_size}_last_return_z',
                f'{asset}_{window_size}_price_zscore',
                f'{asset}_{window_size}_mfe',
                f'{asset}_{window_size}_mae',
                # f'{asset}_{window_size}_tail_variance',
                # f'{asset}_{window_size}_downside_vol',
                f'{asset}_{window_size}_momentum_zscore',
                f'{asset}_{window_size}_return_entropy',
                f'{asset}_{window_size}_variance_ratio',
                # f'{asset}_{window_size}_win_streak',
                # f'{asset}_{window_size}_loss_streak',
                # f'{asset}_{window_size}_effective_bets',
                f'{asset}_{window_size}_mad',
                f'{asset}_{window_size}_gini',
                # f'{asset}_{window_size}_hill_estimator',
                f'{asset}_{window_size}_drawdown_at_risk',
                f'{asset}_{window_size}_cov',
                # f'{asset}_{window_size}_cash_ratio',
                # 'asset_count',
                # f'{asset}_{window_size}_concentration_ratio',
                # f'{asset}_{window_size}_trade_count',
                f'{asset}_{window_size}_asset_value_zscore',
                f'{asset}_{window_size}_avg_win_return',
                f'{asset}_{window_size}_avg_loss_return',
                f'{asset}_{window_size}_max_single_drawdown',
                # f'{asset}_{window_size}_avg_trade_size',
                f'{asset}_{window_size}_realized_pnl',
                f'{asset}_{window_size}_cum_return',
                # 'episode_progress',
                # f'{asset}_{window_size}_sterling_ratio',
                f'{asset}_{window_size}_ulcer_ratio',
                f'{asset}_{window_size}_cdar',
                f'{asset}_{window_size}_avg_holding_period',
                f'{asset}_{window_size}_new_high_count',
                f'{asset}_{window_size}_recovery_ratio',
                f'{asset}_{window_size}_skew_vol_ratio',
                f'{asset}_{window_size}_mean_return',
                f'{asset}_{window_size}_variance',
                f'{asset}_{window_size}_range_return',
                f'{asset}_{window_size}_iqr_return',
                f'{asset}_{window_size}_jarque_bera',
                f'{asset}_{window_size}_autocorr_sq',
                f'{asset}_{window_size}_drawdown_rate',
                f'{asset}_{window_size}_drawdown_entropy',
                f'{asset}_{window_size}_win_loss_ratio',
                f'{asset}_{window_size}_avg_win_streak',
                f'{asset}_{window_size}_avg_loss_streak',
                f'{asset}_{window_size}_tail_return_ratio',
                f'{asset}_{window_size}_spectral_entropy',
                f'{asset}_{window_size}_avg_drawdown',
                f'{asset}_{window_size}_annual_return',
                f'{asset}_{window_size}_annual_volatility',
                f'{asset}_{window_size}_annual_sharpe',
                # f'{asset}_{window_size}_cagr',
                f'{asset}_{window_size}_trend_slope',
                f'{asset}_{window_size}_trend_r2',
                f'{asset}_{window_size}_return_consistency',
                f'{asset}_{window_size}_zero_crossing_rate',
                ]

        self.metrics.update({col: 0 for col in self.portfolio_metrics_col})
        print(f'||| self.fe_dict: {self.fe_dict} ||| self.fe_dict.keys() : {self.fe_dict.keys()}')
        # stop

        self.historical_trades = pd.DataFrame(self.data.loc[:self.end_step+1]['timestamp'])

        self.historical_trades['timestamp'] = pd.to_datetime(self.historical_trades['timestamp'], unit='ns') # 'd')
        self.historical_trades.set_index('timestamp', inplace=True)


        self.check_returns = 0
        self.price = 0
        self.action = 0
        self.reward = 0

        # Additional attributes
        self.returns.clear() # = []  # Initializing as a list
        self.rewards = []  # Initializing as a list
        self.cumulative_returns_history = []  # To track cumulative returns over time
        self.cumulative_history = []

        self.max_step = self.end_step - 1 - self.start_step
        self.max_ep_len = int(self.end_step - self.start_step) # Origianl.

        # Define action space: continuous actions representing proportions of the portfolio
        # self.action_space = gym.spaces.Discrete(n=len(selected_assets)) # low=0, high=1, shape=(len(selected_assets),), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(selected_assets), ), dtype=np.float32) # Original.


        # Ensure the price data has a datetime index.
        if not isinstance(self.price_data.index, pd.DatetimeIndex):
            if 'timestamp' in self.price_data.columns:
                self.price_data.set_index('timestamp', inplace=True)
                logging.info("Price data index set to timestamp.")
            else:
                logging.error("Price data does not contain a 'timestamp' column.")
                raise ValueError("Missing 'timestamp' column in price_data")
                
        # Historical trades is expected to be a DataFrame where we will record buy-hold values.
        # If the historical trades do not have a datetime index, try to set one using the 'timestamp' column.
        if not isinstance(self.historical_trades.index, pd.DatetimeIndex):
            if 'timestamp' in self.historical_trades.columns:
                self.historical_trades.set_index('timestamp', inplace=True)
                logging.info("Historical trades index set to timestamp column.")
            else:
                logging.warning("Historical trades do not have a timestamp column; index alignment may fail.")
       

        self.init_buy_holds(index=self.start_step)

        # Initialize the live PnL calculator.
        self.live_pnl = LivePnLCalculator(self.selected_assets)
        # Initialize dictionary to hold the previous total PnL per asset for trade evaluation.
        self.last_asset_pnl = {asset: 0.0 for asset in self.selected_assets}
        self.method = 'FIFO' # 'LIFO'

        n_assets = len(self.selected_assets)
        self.n_assets = n_assets

        # Create a tracker for block‐probabilities
        self.block_tracker = BlockProbabilityTracker(n_assets, prior=1.0)


        self.init_env_attributes()
        # print(f'historical_trades: {self.historical_trades}')
        self.init_historical_trades(debug=False)
        print(f'historical_trades: {self.historical_trades}')
        self.df_raw = deepcopy(self.historical_trades)
        print(f'self.df_raw: {self.df_raw}')


        if self.use_2d:
            # self.state, self.obs = self.get_state()
            obs_, obs = self.reset()
        else:
            self.initialized = False # Original.
            # obs = self.get_state()
            obs = self.reset(debug=False)

        print(f'historical_trades: {self.historical_trades}')

        
        if self.use_2d:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[0], obs.shape[1], obs.shape[2]), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float32) # shape=(window_size * self.num_cols,)
            # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(window_size * self.num_cols,), dtype=np.float32)



    def init_buy_holds(self, index: int):
        """
        Calculate the initial 'buy-hold' values for each asset (BTC, ETH, SP500)
        based on the price at a given index in self.price_data. The buy-holds
        represent what the portfolio value would be if you had bought the asset
        with your entire initial balance at the given price.
        
        This method also initializes various trade/PnL columns in self.historical_trades.
        """
        # --- Ensure both indexes are datetime and normalized ---
        try:
            # Remove any timezone (normalize to naive datetime)
            self.price_data.index = pd.to_datetime(self.price_data.index).tz_localize(None)
            self.historical_trades.index = pd.to_datetime(self.historical_trades.index).tz_localize(None)
        except Exception as e:
            logging.error("Error converting indexes to datetime: %s", str(e))
            raise

        # --- Calculate the initial volumes based on the given price_data row ---
        try:
            eth_init_volume = self.initial_balance / self.price_data.iloc[index]['ETH Close']
            btc_init_volume = self.initial_balance / self.price_data.iloc[index]['Close']
            sp_500_init_volume = self.initial_balance / self.price_data.iloc[index]['SP500 Close']
            # logging.info("Calculated initial volumes: BTC=%.4f, ETH=%.4f, SP500=%.4f",
            #              btc_init_volume, eth_init_volume, sp_500_init_volume)
        except Exception as e:
            logging.error("Error calculating initial volumes: %s", str(e))
            raise

        # --- Reindex the price series to the historical_trades index using forward-fill ---
        try:
            # Using method='ffill' ensures that if an exact match is not found,
            # the last available price is used.
            new_cols = {}
            new_cols['Close_buy_hold'] = self.price_data['Close'].reindex(self.historical_trades.index, method='ffill') * btc_init_volume
            new_cols['ETH Close_buy_hold'] = self.price_data['ETH Close'].reindex(self.historical_trades.index, method='ffill') * eth_init_volume
            new_cols['SP500 Close_buy_hold'] = self.price_data['SP500 Close'].reindex(self.historical_trades.index, method='ffill') * sp_500_init_volume
        except Exception as e:
            logging.error("Error reindexing price series: %s", str(e))
            raise



    def record_historical_data(self, metrics, debug=False):
        # Ensure dtype compatibility for numerical columns
        float_cols = [
            col for col in self.historical_trades.columns
            if '_action' in col or '_values' in col or '_weights' in col
            or 'quantity' in col or 'buy_hold' in col or 'time_span' in col or 'cash' in col
            # and not 'cash_quantity' or 'cash_buy_hold'
        ]
        self.historical_trades[float_cols] = self.historical_trades[float_cols].astype(float)

        # Ensure timestamp exists before modifying
        if self.timestamp not in self.historical_trades.index:
            print(f'||| self.timestamp: {self.timestamp} ||| self.historical_trades.index: {self.historical_trades.index} ||| self.current_step: {self.current_step}')
            # self.historical_trades.loc[self.timestamp] = None  # Initialize row if missing
            self.historical_trades.reset_index('timestamp', drop=False, inplace=True)
            print(f'||| self.timestamp: {self.timestamp} ||| self.historical_trades.index: {self.historical_trades.index}')

        self.historical_trades.at[self.timestamp, 'cash_balance'] = float(self.balance)
        self.historical_trades.at[self.timestamp, 'previous_cash_balance'] = float(self.historical_trades.at[self.previous_timestamp, 'cash_balance'])


        # Retrieve current weights
        current_weights = self.get_current_weights  # Ensure this returns a valid list
        hrp_current_weights = self.hrp_get_current_weights  # Ensure this returns a valid list
        current_prices = self.get_current_prices_dict()
        previous_weights = self.previous_weights

        p_block = self.block_tracker.posterior_mean()
        p_variance = self.block_tracker.posterior_variance()

        turn_over_rates = [np.abs(a - b) for a, b in zip(current_weights, self.previous_weights)] #  turnover_accum += 0.5 * np.abs(action - prev_weights).sum()
        self.historical_trades.at[self.timestamp, "portfolio_turn_over_rate"] = float(np.sum(turn_over_rates))
        self.historical_trades.at[self.timestamp, f'should_be_portfolio_values'] = float(self.should_be_portfolio_value)

        self.historical_trades.at[self.timestamp, f'Portfolio_Dirchlet_Posterior_entropy_getting_blocked'] = float(self.block_tracker.posterior_entropy())
        for asset, i in zip(self.portfolio.keys(), range(len(current_weights))):

            self.historical_trades.at[self.timestamp, f'{asset}_turn_over_rate'] = float(turn_over_rates[i])
            asset_value = float(self.portfolio_value * current_weights[i])
            self.historical_trades.at[self.timestamp, f'{asset}_values'] = asset_value # float(self.portfolio_value * current_weights[i])
            previous_asset_value = float(self.previous_portfolio_value * previous_weights[i])
            self.historical_trades.at[self.timestamp, f'{asset}_previous_values'] = previous_asset_value
            self.historical_trades.at[self.timestamp, f'{asset}_delta_values'] = float(float(asset_value - previous_asset_value) / previous_asset_value) if previous_asset_value != 0. else 0.
            self.historical_trades.at[self.timestamp, f'{asset}_weights'] = float(current_weights[i])
            self.historical_trades.at[self.timestamp, f'{asset}_previous_weights'] = float(self.previous_weights[i])
            self.historical_trades.at[self.timestamp, f'{asset}_delta_weights'] = float(float(current_weights[i]) - float(previous_weights[i]))
            self.historical_trades.at[self.timestamp, f'portfolio_miuns_{asset}_values'] = float(self.portfolio_value - asset_value)


            hrp_asset_value = float(self.hrp_portfolio_value * hrp_current_weights[i])
            self.historical_trades.at[self.timestamp, f'hrp_{asset}_values'] = hrp_asset_value # Original and Good.               # float(self.portfolio_value * current_weights[i])
            self.historical_trades.at[self.timestamp, f'hrp_{asset}_weights'] = float(float(hrp_current_weights[i]) - float(current_weights[i]))
            self.historical_trades.at[self.timestamp, f'hrp_portfolio_miuns_{asset}_values'] = float(
                hrp_asset_value - asset_value
                ) / asset_value if asset_value != 0. else 0. # float(float(hrp_asset_value - asset_value) / 

            if i != 0:
                self.historical_trades.at[self.timestamp, f'{asset}_Dirchlet_Posterior_mu_getting_blocked'] = float(p_block[i-1])
                self.historical_trades.at[self.timestamp, f'{asset}_Dirchlet_Posterior_variance_getting_blocked'] = float(p_variance[i-1])

                if self.use_action_norm:
                    self.historical_trades.at[self.timestamp, f'{asset}_trade_action'] = float(self.actions[i-1])
                    self.historical_trades.at[self.timestamp, f'{asset}_intended_trade_weight'] = float(self.action_weights[i-1])
                self.historical_trades.at[self.timestamp, f'hrp_{asset}_action'] = float(float(self.hrp_actions[i-1]) - float(self.actions[i-1]))

                if asset == 'Close':
                    self.historical_trades.at[self.timestamp, f'{asset}_buy_hold_minus_portfolio'] = float(float(self.btc_buy_hold - self.btc_value) / self.btc_buy_hold) if self.btc_value != 0. else 0.
                    self.historical_trades.at[self.timestamp, f'hrp_portfolio_{asset}_owned'] = self.hrp_portfolio.get(f'{asset}', 0.0)
                    if self.hrp_btc_value == 0:
                        self.historical_trades.at[self.timestamp, f'hrp_{asset}_buy_hold_minus_portfolio'] = float(0.)
                    else:
                        self.historical_trades.at[self.timestamp, f'hrp_{asset}_buy_hold_minus_portfolio'] = float(float(self.hrp_btc_value - self.btc_value) / self.hrp_btc_value) if self.hrp_btc_value != 0. else 0.
                if asset == 'ETH Close':
                    self.historical_trades.at[self.timestamp, f'{asset}_buy_hold_minus_portfolio'] = float(float(self.eth_buy_hold - self.eth_value) / self.eth_buy_hold) if self.eth_value != 0. else 0.
                    self.historical_trades.at[self.timestamp, f'hrp_portfolio_{asset}_owned'] = self.hrp_portfolio.get(f'{asset}', 0.0)
                    if self.hrp_eth_value == 0:
                        self.historical_trades.at[self.timestamp, f'hrp_{asset}_buy_hold_minus_portfolio'] = float(0.)
                    else:
                        self.historical_trades.at[self.timestamp, f'hrp_{asset}_buy_hold_minus_portfolio'] = float(float(self.hrp_eth_value - self.eth_value) / self.hrp_eth_value) if self.hrp_eth_value != 0. else 0. # / self.hrp_eth_value)
                if asset == 'SP500 Close':
                    self.historical_trades.at[self.timestamp, f'{asset}_buy_hold_minus_portfolio'] = float(float(self.sp500_buy_hold - self.sp500_value) / self.sp500_buy_hold) if self.sp500_buy_hold != 0. else 0.
                    self.historical_trades.at[self.timestamp, f'hrp_portfolio_{asset}_owned'] = self.hrp_portfolio.get(f'{asset}', 0.0) # float(self.hrp_sp500_value) # hrp_prev_asset_quantity_list
                    if self.hrp_sp500_value == 0:
                        self.historical_trades.at[self.timestamp, f'hrp_{asset}_buy_hold_minus_portfolio'] = float(0.)
                    else:
                        self.historical_trades.at[
                            self.timestamp,
                            f'hrp_{asset}_buy_hold_minus_portfolio'
                            ] = float(float(self.sp500_value - self.hrp_sp500_value) / self.hrp_sp500_value)  if self.hrp_sp500_value != 0. else 0.




        # Assign close quantities
        self.historical_trades.at[self.timestamp, 'portfolio_Close_owned'] = float(self.btc_quantity)
        self.historical_trades.at[self.timestamp, 'portfolio_ETH Close_owned'] = float(self.eth_quantity)
        self.historical_trades.at[self.timestamp, 'portfolio_SP500 Close_owned'] = float(self.sp500_quantity)

        # Assign close quantities
        self.historical_trades.at[self.timestamp, 'hrp_Close_quantity'] = float(float(self.hrp_portfolio.get('Close', 0.0)) - float(self.btc_quantity)) # float(float(self.hrp_btc_quantity) - float(self.btc_quantity))
        self.historical_trades.at[self.timestamp, 'hrp_ETH Close_quantity'] = float(float(self.hrp_portfolio.get('ETH Close', 0.0)) - float(self.eth_quantity)) # float(float(self.hrp_eth_quantity) - float(self.eth_quantity))
        self.historical_trades.at[self.timestamp, 'hrp_SP500 Close_quantity'] = float(float(self.hrp_portfolio.get('SP500 Close', 0.0)) - float(self.sp500_quantity)) # float(float(self.hrp_sp500_quantity) - float(self.sp500_quantity))

        self.historical_trades.at[self.timestamp, 'var'] = float(self.var)
        self.historical_trades.at[self.timestamp, 'max_drawdown'] = float(self.max_drawdown)
        self.historical_trades.at[self.timestamp, 'parametric_var'] = float(self.parametric_var)
        self.historical_trades.at[self.timestamp, 'parametric_var_dollar'] = float(self.parametric_var_dollar)

        self.historical_trades.at[self.timestamp, 'cvar'] = float(self.cvar)
        self.historical_trades.at[self.timestamp, 'parametric_cvar'] = float(self.parametric_cvar)
        self.historical_trades.at[self.timestamp, 'cvar_dollar'] = float(self.cvar_dollar)
        self.historical_trades.at[self.timestamp, 'profit'] = float(self.profit)
        self.historical_trades.at[self.timestamp, 'return_'] = float(self.return_)

        self.historical_trades.at[self.timestamp, 'previous_profit'] = float(self.previous_profit)
        self.historical_trades.at[self.timestamp, 'prev_var'] = float(self.prev_var)

        self.historical_trades.at[self.timestamp, 'prev_parametric_var'] = float(self.prev_parametric_var)
        self.historical_trades.at[self.timestamp, 'prev_parametric_cvar'] = float(self.prev_parametric_cvar)
        self.historical_trades.at[self.timestamp, 'prev_cvar_dollar'] = float(self.prev_cvar_dollar)
        self.historical_trades.at[self.timestamp, 'prev_return_'] = float(self.prev_return_)
        self.historical_trades.at[self.timestamp, 'prev_parametric_var_dollar'] = float(self.prev_parametric_var_dollar)

        self.historical_trades.at[self.timestamp, 'portfolio_value'] = float(self.portfolio_value)
        self.historical_trades.at[self.timestamp, 'hrp_portfolio_value'] = float(self.hrp_portfolio_value)
        if self.hrp_portfolio_value != 0:
            self.historical_trades.at[self.timestamp, 'portfolio_value_miuns_hrp_portfolio_value'] = float(float(self.portfolio_value - self.hrp_portfolio_value) / self.hrp_portfolio_value)
        else:
            self.historical_trades.at[self.timestamp, 'portfolio_value_miuns_hrp_portfolio_value'] = float(0.)


        self.engine.record(self.portfolio_value, market_return=self.market_return, risk_free_rate=self.risk_free_rate_) # Original.

        features = self.engine.get_latest_features()

        self.historical_trades.at[self.timestamp, "pf_alpha_monthly"] = features["pf_alpha_monthly"]
        self.historical_trades.at[self.timestamp, "pf_vol_monthly"] = features["pf_vol_monthly"]

        self.historical_trades.at[self.timestamp, "pf_sharpe"] = features["pf_sharpe"]
        self.historical_trades.at[self.timestamp, "pf_sortino"] = features["pf_sortino"]
        self.historical_trades.at[self.timestamp, "pf_max_drawdown"] = features["pf_max_drawdown"]
        self.historical_trades.at[self.timestamp, "pf_beta"] = features["pf_beta"]
        self.historical_trades.at[self.timestamp, "pf_beta_adj_sharpe"] = features["pf_beta_adj_sharpe"]
        self.historical_trades.at[self.timestamp, "pf_m_squared_ratio"] = features["pf_m_squared_ratio"]


        for col in self.portfolio_metrics_col:

            self.historical_trades.at[self.timestamp, col] = metrics.get(f'{col}', 0.0)


        if debug:
            row = self.historical_trades.loc[self.timestamp]
            print("\n--- Value Check for row @", self.timestamp, "---")
            for col, val in row.items():
                if pd.isna(val):
                    print(f"[NaN] {col} = {val}")
                elif np.isinf(val):
                    print(f"[Inf] {col} = {val}")
                elif val == 0:
                    print(f"[Zero] {col} = {val}")





    def get_current_prices_dict(self) -> dict:
        try:
            # current_timestamp = self.price_data.index[self.current_step]
            prices = self.price_data.loc[self.timestamp].to_dict()
            # logger.debug(
            #     "Current prices at %s: %s", self.timestamp, prices
            #     )
            return prices
        except Exception as e:
            logger.error(
                "Error obtaining current prices at step %d: %s", self.current_step, str(e)
                )
            raise


    def get_previous_prices_dict(self) -> dict:
        try:
            # current_timestamp = self.price_data.index[self.current_step]
            prices = self.price_data.loc[self.previous_timestamp].to_dict()
            # logger.debug(
            #     "Current prices at %s: %s", self.timestamp, prices
            #     )
            return prices
        except Exception as e:
            logger.error(
                "Error obtaining current prices at step %d: %s", self.current_step, str(e)
                )
            raise

    def get_next_prices_dict(self) -> dict:
        try:
            # current_timestamp = self.price_data.index[self.current_step]
            prices = self.price_data.loc[self.next_timestamp].to_dict()
            # logger.debug(
            #     "Current prices at %s: %s", self.timestamp, prices
            #     )
            return prices
        except Exception as e:
            logger.error(
                "Error obtaining current prices at step %d: %s", self.current_step, str(e)
                )
            raise


    def evaluate_asset_trade(
        self, asset: str, current_price: float,
        timestamp: pd.Timestamp
        ) -> dict:
        """
        Evaluate the asset-specific trade by comparing the new total PnL
        (realized + unrealized) with the previous stored value.
        Returns a dictionary with delta values.
        """
        try:
            current_prices = self.get_current_prices()
            trade_price=current_prices[asset]
            fifo_r, fifo_u, lifo_r, lifo_u = self.live_pnl.update_asset_pnl(asset, current_price, timestamp)
            fifo_r = self.live_pnl.get_realized_pnl_fifo(asset)
            fifo_u = self.live_pnl.get_unrealized_pnl_fifo(asset, current_price=trade_price)
            lifo_r = self.live_pnl.get_realized_pnl_lifo(asset)
            lifo_u = self.live_pnl.get_unrealized_pnl_lifo(asset, current_price=trade_price)
            total_pnl = fifo_r + fifo_u  # You may choose one method (FIFO or LIFO) or compare both.
            previous = self.last_asset_pnl.get(asset, 0.0)
            delta = total_pnl - previous
            # Log the evaluation:
            logger.debug("Trade evaluation for %s at %s: delta PnL=%.4f (previous=%.4f, current=%.4f)",
                         asset, timestamp, delta, previous, total_pnl)
            # Update the stored pnl for future comparisons.
            self.last_asset_pnl[asset] = total_pnl
            return {"delta_pnl": delta}
        except Exception as e:
            logger.error("Error evaluating trade for %s: %s", asset, str(e))
            return {"delta_pnl": None}




    def sharpe_ratio(self) -> float:
        r = np.array(self.returns)
        if r.size < 2:
            return 0.0
        mean = r.mean()
        std = r.std() if r.std() > 1e-8 else 0.
        return mean / std if std > 0 else 0.0

    def sortino_ratio(self) -> float:
        r = np.array(self.returns)
        if r.size < 2:
            return 0.0
        mean = r.mean()
        neg = r[r < 0]
        downside_std = neg.std() if neg.size > 0 and neg.std() > 1e-8 else 0.0
        return mean / downside_std if downside_std > 0 else 0.0

    def cvar(self) -> float:
        r = np.array(self.returns)
        if r.size < 2:
            return 0.0
        sorted_r = np.sort(r)
        k = max(int(self.cvar_q * sorted_r.size), 1)
        return -sorted_r[:k].mean()

    def entropy_bonus(self, weights: np.ndarray) -> float:
        """
        Encourages diversification by adding entropy of the weight vector.
        """
        w = np.clip(weights, 1e-8, 1.0)
        w /= w.sum()
        return -np.sum(w * np.log(w))



    @staticmethod
    def hurst_exponent(r: np.ndarray) -> float:
        """Estimate Hurst exponent via R/S analysis for 0 < H < 1."""
        N = len(r)
        if N < 2:
            return 0.5
        mean_r = np.mean(r)
        Y = np.cumsum(r - mean_r)
        R = np.max(Y) - np.min(Y)
        S = np.std(r, ddof=1) if np.std(r, ddof=1) > 1e-8 else 0.
        if S == 0 or R == 0:
            return 0.5
        return float(np.log(R / S) / np.log(N))


    def update(
        self,
        portfolio_value: float,
        weights: np.ndarray,
        benchmark_return: Optional[float] = None,
        debug: bool = False
    ):
        """
        Call this method once per step to record the latest portfolio state.
        
        :param portfolio_value: Current total portfolio value.
        :param weights: Current portfolio weights vector (must sum to 1).
        :param benchmark_return: Optional single-step benchmark return.
        """
        # Initialize on first call
        # if not self.portfolio_values: # Original but not working.
        if self.current_step == self.start_step:
            self._peak_value = portfolio_value
            self._peak_index = 0
            self._current_drawdown_start = 0
            self.portfolio_values.append(portfolio_value)
            self.weights_history.append(weights.copy())
            if benchmark_return is not None:
                self.benchmark_returns.append(benchmark_return)
            self.drawdowns.append(0.0)
            self.vol_history.append(0.0)
            self.returns.append(0.0)
            self.prev_ema_short = portfolio_value
            self.prev_ema_long = portfolio_value
            return # Original.
        
        # Compute and store return
        # prev_value = self.portfolio_values[-1] # Original.
        # ret = portfolio_value / prev_value - 1 # Original.
        current_prices = self.get_current_prices_list()
        previous_prices = self.get_previous_prices_list()
        # print(f'||| current_prices: {current_prices} ||| previous_prices: {previous_prices} |||')
        previous_weights = self.previous_weights[1:]
        current_weights = self.get_current_weights[1:]
        # delta = [float(float(p_t - prev_p) / prev_p) for p_t, prev_p in zip(current_prices, previous_prices)]

        prev_asset_quantity_list = []
        for asset in self.selected_assets:
            prev_asset_quantity_list.append(float(self.historical_trades.at[self.previous_timestamp, f'portfolio_{asset}_owned']))

        # should_be_portfolio_value = self.prev_balance + sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, previous_prices)])
        should_be_portfolio_value = self.prev_balance + np.sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, current_prices)])
        # ret = (self.portfolio_value - should_be_portfolio_value) / should_be_portfolio_value if should_be_portfolio_value != 0. else float(self.portfolio_value - should_be_portfolio_value) / self.previous_portfolio_value
        ret = float(self.portfolio_value - should_be_portfolio_value) / self.previous_portfolio_value if self.previous_portfolio_value != 0. else float(self.portfolio_value - should_be_portfolio_value) / should_be_portfolio_value
        self.returns.append(float(ret)) # Original.
        if len(self.returns) > self.seq_len:
            self.returns.pop(0)
        
        # Store benchmark return
        if benchmark_return is not None:
            self.benchmark_returns.append(benchmark_return)
            if len(self.benchmark_returns) > self.seq_len:
                self.benchmark_returns.pop(0)
        
        # Update portfolio value history
        self.portfolio_values.append(portfolio_value) # Not Already happened in the step function.

        # if len(self.price_history) > self.momentum_window + 1:
        if len(self.portfolio_values) > self.momentum_window + 1:
            self.portfolio_values.pop(0)
        # Compute and store price momentum
        # if len(self.price_history) > self.momentum_window:
        if len(self.portfolio_values) > self.momentum_window:
            entry_price = self.portfolio_values[-(self.momentum_window + 1)]
            price_mom = portfolio_value / entry_price - 1 if entry_price != 0. else 0.
        else:
            price_mom = 0.0
        self.momentum_history.append(price_mom)
        if len(self.momentum_history) > self.seq_len:
            self.momentum_history.pop(0)
        
        # Update drawdown tracking
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._peak_index = len(self.portfolio_values) - 1
            self._current_drawdown_start = self._peak_index
        else:
            drawdown = (
                self._peak_value - portfolio_value
                ) / self._peak_value if self._peak_value != 0. else 0.
            dd_length = len(
                self.portfolio_values
                ) - self._current_drawdown_start

            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
            # self._max_drawdown = max(self._max_drawdown, drawdown)

            if dd_length > self._max_drawdown_length:
                self._max_drawdown_length = dd_length
            # self._max_drawdown_length = max(self._max_drawdown_length, dd_length)

        self.drawdowns.append(self._max_drawdown)
        if len(self.drawdowns) > self.seq_len:
            self.drawdowns.pop(0)


        # Compute rolling volatility (std of returns)
        vol = float(np.std(self.returns)) if len(self.returns) > 0 and np.std(self.returns) > 1e-8 else 0.0
        self.vol_history.append(vol)
        if len(self.vol_history) > self.seq_len:
            self.vol_history.pop(0)
        
        # Store weights for turnover calculation
        self.weights_history.append(weights.copy())
        if len(self.weights_history) > self.seq_len + 1:
            self.weights_history.pop(0)

        # Update EMAs
        self.prev_ema_short = (
            self.alpha_short * portfolio_value +
            (1 - self.alpha_short) * self.prev_ema_short
        )
        self.prev_ema_long = (
            self.alpha_long * portfolio_value +
            (1 - self.alpha_long) * self.prev_ema_long
        )
        if debug:
            print(
f'''
||| portfolio_value: {portfolio_value}
||| self.alpha_long: {self.alpha_long}
||| self.prev_ema_long: {self.prev_ema_long}
||| self.alpha_short: {self.alpha_short}
||| self.prev_ema_short: {self.prev_ema_short}
'''
            )
            # stop

        # 1. Cash ratio
        cash = float(self.portfolio.get('cash', 0.0))
        cash_ratio = cash / portfolio_value if portfolio_value > 0 else 0.0
        self.cash_ratios.append(cash_ratio)
        if len(self.cash_ratios) > self.seq_len:
            self.cash_ratios.pop(0)

        # 2. Asset count (non-zero holdings excluding cash)
        asset_count = float(
            sum(
                1 for k, v in self.portfolio.items() if k != 'cash' and v > 0
                )
            )
        self.asset_counts.append(asset_count)
        if len(self.asset_counts) > self.seq_len:
            self.asset_counts.pop(0)

        # 3. Concentration ratio (sum of two largest weights)
        sorted_w = np.sort(weights)[::-1]
        conc_ratio = float(np.sum(sorted_w[:2]))
        self.concentration_ratios.append(conc_ratio)
        if len(self.concentration_ratios) > self.seq_len:
            self.concentration_ratios.pop(0)

        # 4. Trade count: did a trade occur this step?
        if len(self.weights_history) > 1:
            prev_w = self.weights_history[-2]
            diff = float(
                np.sum(
                    np.abs(weights - prev_w)
                    )
                )
            # traded = 1.0 if diff > self.min_trade_value else 0.0
            traded = 1.0 if diff != 0. else 0.0
        else:
            traded = 0.0
        # self.trade_counts.append(self.traded)
        self.trade_counts.append(traded)
        if len(self.trade_counts) > self.seq_len:
            self.trade_counts.pop(0)

        # 5. Portfolio value Z-score over recent window
        vals = np.array(self.portfolio_values[-self.seq_len:], dtype=np.float32)
        if vals.size > 1:
            mean_v = vals.mean()
            std_v = vals.std() if len(self.returns) > 0 and vals.std() > 1e-8 else 0.0
            vz = float(
                (portfolio_value - mean_v) / std_v
                ) if std_v > 0 else 0.0
        else:
            vz = 0.0
        self.value_zscores.append(vz)
        if len(self.value_zscores) > self.seq_len:
            self.value_zscores.pop(0)

        # Realized PnL
        pv_list = self.portfolio_values
        if len(pv_list) > 1:
            pnl = pv_list[-1] - pv_list[-2]
        else:
            pnl = 0.0
        self.realized_pnls.append(pnl)
        if len(self.realized_pnls) > self.seq_len:
            self.realized_pnls.pop(0)

        # Cumulative return
        initial = pv_list[0] if pv_list else portfolio_value
        cum_ret = (portfolio_value / initial - 1) if initial != 0 else 0.0
        self.cumulative_returns.append(cum_ret)
        if len(self.cumulative_returns) > self.seq_len:
            self.cumulative_returns.pop(0)

    def compute_metrics(self, debug=False) -> Dict[str, float]:
        """
        Compute and return all configured financial metrics.
        """
        metrics: Dict[str, float] = {}
        r = np.array(self.returns)
        
        # Basic stats
        mean_r = r.mean() if r.size > 0 else 0.0
        std_r = r.std() if r.size > 0 and r.std() > 1e-8 else 0.0
        
        # Sharpe ratio
        excess = r - self.risk_free_rate_
        metrics['sharpe'] = (excess.mean() / std_r) if std_r > 0 else 0.0
        
        # Sortino ratio
        neg_r = r[r < self.risk_free_rate_]
        downside_std = neg_r.std() if neg_r.size > 0 and neg_r.std() > 1e-8 else 0.0
        metrics['sortino'] = float(
            excess.mean() / downside_std) if downside_std > 0 else 0.0

        # Momentum (cumulative return)
        metrics['momentum'] = float(np.prod(1 + r) - 1) if r.size > 0 else 0.0

        # Rolling skewness & kurtosis
        if std_r > 1e-8:
            skewness = float(np.mean((r - mean_r)**3) / (std_r**3))
            kurtosis = float(np.mean((r - mean_r)**4) / (std_r**4) - 3.0)
            metrics['skewness'] = skewness
            metrics['kurtosis'] = kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
            metrics['skewness'] = 0.0
            metrics['kurtosis'] = 0.0

        # Realized volatility annualized (assumes daily steps)
        realized_vol = std_r * np.sqrt(252) if std_r > 1e-8 else 0.0
        metrics['realized_vol'] = realized_vol

        # Ulcer Index: sqrt of mean squared drawdowns
        dd_arr = np.array(self.drawdowns)
        ulcer_index = float(np.sqrt(np.mean(dd_arr**2))) if dd_arr.size > 0 or np.mean(dd_arr**2) > 0. else 0.0
        metrics['ulcer_index'] = ulcer_index

        # Volatility of volatility
        vh = np.array(self.vol_history)
        metrics['vol_of_vol'] = float(np.std(vh)) if vh.size > 0 and np.std(vh) > 1e-8 else 0.0

        """# Beta & correlation with benchmark if available
        if self.benchmark_returns:
            br = np.array(self.benchmark_returns)
            if debug:
                print(
f'''
||| self.benchmark_returns: {self.benchmark_returns}
||| r: {r}
||| len(r): {len(r)}
||| br: {br}
||| len(br): {len(br)}
'''
                 )
            cov = float(np.cov(r, br, ddof=0)[0, 1]) if br.size >=3 else 0.0
            # cov = safe_cov(data, rowvar=False, ddof=1)
            var_b = float(np.var(br)) if br.size > 0 else 0.0
            metrics['beta'] = cov / var_b if var_b != 0 else 0.0
            correlation = float(np.corrcoef(r, br)[0, 1]) if br.size > 1 else 0.0
            if not np.isnan(correlation) and not np.isinf(correlation):
                metrics['correlation'] = correlation if br.size > 1 else 0.0 # float(np.corrcoef(r, br)[0, 1]) if br.size > 1 else 0.0
            else:
                metrics['correlation'] = 0.0
        else:
            # metrics['beta'] = 0.0
            metrics['correlation'] = 0.0"""



        # Beta & correlation with benchmark if available
        if self.benchmark_returns:
            r = np.asarray(self.returns, dtype=float)
            br = np.asarray(self.benchmark_returns, dtype=float)

            # 1) covariance and variance via safe_cov
            cov_rb = safe_cov(r, br, ddof=0)
            var_b = safe_cov(br, ddof=0)

            if debug:
                print(
f'''
||| self.benchmark_returns: {self.benchmark_returns}
||| r: {r}
||| len(r): {len(r)}
||| br: {br}
||| len(br): {len(br)}
'''
                 )

            # 2) beta
            metrics[f'beta'] = cov_rb / var_b if var_b != 0.0 else 0.0

            try:
                correlation = float(np.corrcoef(r, br)[0, 1]) if br.size > 1 else 0.0
            except RuntimeWarning as e:
                print("Caught invalid‐divide warning:", e)
                # fallback to safe routine
                # correlation = float(safe_corrcoef(r, br)[0, 1])
                correlation = 0. # float(np.corrcoef(r, br)[0, 1]) if br.size > 1 else 0.0

            metrics['correlation'] = correlation

            # 3) correlation
            var_r = safe_cov(r, ddof=0)
            # if var_r > 0.0 and var_b > 0.0:
            #     metrics[f'{self.asset}_{self.window}_correlation'] = cov_rb / (math.sqrt(var_r) * math.sqrt(var_b))
            #     metrics[f'{self.asset}_{self.window}_correlation'] = correlation
            # else:
            #     metrics[f'{self.asset}_{self.window}_correlation'] = 0.0
        else:
            metrics['correlation'] = 0.0
            metrics['beta'] = 0.0


        
        # Omega ratio (threshold at risk-free rate)
        gains = r[r >= self.risk_free_rate_] - self.risk_free_rate_
        losses = self.risk_free_rate_ - r[r < self.risk_free_rate_]
        sum_gains = gains.sum() if gains.size > 0 else 0.0
        sum_losses = losses.sum() if losses.size > 0 else 1.0
        metrics['omega'] = sum_gains / sum_losses
        
        # Calmar ratio
        metrics['calmar_mean_r'] = (mean_r / self._max_drawdown) if self._max_drawdown != 0 else 0.0
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        if r.size > 0:
            var_level = np.percentile(r, 100 * 0.05) # self.var_percentile)
            metrics['var'] = -var_level
            metrics['cvar'] = -r[r <= var_level].mean() if r[r <= var_level].size > 0 else 0.0
        else:
            metrics['var'] = 0.0
            metrics['cvar'] = 0.0
        
        # Maximum Drawdown (depth) and Length
        metrics['max_drawdown'] = self._max_drawdown
        metrics['max_drawdown_length'] = float(
            self._max_drawdown_length
            )
        
        # Turnover and Transaction Cost
        if len(self.weights_history) > 1:
            prev_w = self.weights_history[-2]
            curr_w = self.weights_history[-1]
            turnover = float(np.abs(curr_w - prev_w).sum())
            # cost = turnover * self.transaction_cost_rate
            cost = turnover * self.total_transaction_fee
        else:
            turnover, cost = 0.0, 0.0
        # metrics['turnover'] = turnover
        # metrics['transaction_cost'] = cost
        
        # Entropy bonus for diversification
        w = np.clip(self.weights_history[-1], 1e-8, 1.0)
        w /= w.sum()
        metrics['entropy'] = -float(np.sum(w * np.log(w)))
        
        # Risk-parity deviation (distance from equal-weight)
        n = w.size
        target = np.ones(n) / n
        metrics['risk_parity_deviation'] = float(
            np.linalg.norm(w - target)
            )
        
        # Benchmark-relative and Information Ratio
        if self.benchmark_returns:
            br = np.array(self.benchmark_returns)
            active = r - br
            trk_err = active.std() if active.size > 0 and active.std() > 1e-8 else 0.0 # tracking error.
            # info_std = float(active.std()) if active.size > 0 else 0.0
            metrics['excess_return'] = float(mean_r - br.mean())
            metrics['tracking_error'] = float(trk_err)
            metrics['information_ratio'] = float(
                metrics['excess_return'] / trk_err
                ) if trk_err > 0 else 0.0
        else:
            metrics['excess_return'] = 0.0
            metrics['tracking_error'] = 0.0
            metrics['information_ratio'] = 0.0

        # EMA & MACD
        metrics['ema_short'] = float(self.prev_ema_short)
        metrics['ema_long'] = float(self.prev_ema_long)
        metrics['macd'] = metrics['ema_short'] - metrics['ema_long']

        # RSI
        if len(self.returns) >= self.rsi_window:
            recent = r[-self.rsi_window:]
            gains = recent[recent > 0]
            losses = -recent[recent < 0]
            avg_gain = float(np.mean(gains)) if gains.size > 0 else 0.0
            avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
            metrics['rsi'] = float(100 * avg_gain / (avg_gain + avg_loss)) if (avg_gain + avg_loss) != 0 else 50.0
        else:
            metrics['rsi'] = 50.0

        # Win rate & Profit factor
        metrics['win_rate'] = float((r > 0).sum() / r.size) if r.size > 0 else 0.0
        pos = float(np.sum(r[r > 0])) if r.size > 0 else 0.0
        neg = float(-np.sum(r[r < 0])) if r.size > 0 else 0.0
        metrics['profit_factor'] = float(pos / neg) if neg > 0 else 0 # np.inf

        # Rachev ratio
        '''if r.size > 0:
            # k = max(int(self.rachev_percentile * r.size), 1)
            k = max(int(0.05 * r.size), 1)
            
            sorted_r = np.sort(r)
            tail_up = sorted_r[-k:]
            tail_down = sorted_r[:k]
            metrics['rachev'] = float(
                np.mean(tail_up) / -np.mean(tail_down)
                ) if np.mean(tail_down) != 0 else 0.0
        else:
            metrics['rachev'] = 0.0'''

        # Calmar ratio
        cum_return = float(np.prod(1 + r) - 1) if r.size > 0 else 0.0
        max_dd = float(np.max(dd_arr)) if dd_arr.size > 0 else 0.0
        metrics['calmar_cum_r'] = float(cum_return / max_dd) if max_dd != 0 else 0.0

        # Drawdown length (consecutive)
        # length = 0
        # Drawdown recovery time: consecutive steps since last drawdown cleared
        recovery = 0
        dd = np.array(self.drawdowns, dtype=float)
        for d in dd_arr[::-1]:
            if d > 0:
                recovery += 1
            else:
                break
        # metrics['drawdown_length_consecutive'] = float(length)
        metrics['drawdown_recovery_time'] = float(recovery)

        # Autocorrelation lag 1
        if r.size > 3:
            # autocorr_lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if br.size > 1 else 0.0
            try:
                autocorr_lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if br.size > 3 else 0.0
            except RuntimeWarning as e:
                print("Caught invalid‐divide warning:", e)
                # fallback to safe routine
                autocorr_lag1 = float(0.) # np.corrcoef(r[:-1], r[1:])[0, 1]) if br.size > 1 else 0.0 # safe_corrcoef(r[:-1], r[1:])[0, 1] # safe_corrcoef(r, br)[0, 1]
            if not np.isnan(autocorr_lag1) and not np.isinf(autocorr_lag1):
                metrics['autocorr_lag1'] = autocorr_lag1 # float(np.corrcoef(r[:-1], r[1:])[0, 1])
            else:
                metrics['autocorr_lag1'] = 0.0
        else:
            metrics['autocorr_lag1'] = 0.0

        # Hurst exponent
        metrics['hurst'] = self.hurst_exponent(r)

        # Efficiency ratio: net change / sum of absolute changes
        if r.size > 0:
            net = abs(np.sum(r))
            tot = np.sum(np.abs(r))
            metrics['efficiency_ratio'] = float(net / tot) if tot != 0 else 0.0
        else:
            metrics['efficiency_ratio'] = 0.0

        # Kelly fraction: mean / variance
        mean_r = float(np.mean(r)) if r.size > 0 else 0.0
        var_r = float(np.var(r)) if r.size > 0 and np.var(r) > 1e-8 else 1.0 # 0.0 # 1.0
        # print(f'||| mean_r: {mean_r} ||| var_r: {var_r} ||| r: {r} |||')
        metrics['kelly_fraction'] = float(mean_r / var_r) if var_r > 1e-8 else float(0.)

        # Last return Z-score
        if r.size > 1:
            if r.size > 0:
                metrics['last_return_z'] = float(
                    (r[-1] - mean_r) / (
                        np.std(r) if np.std(r) > 1e-8 else 1
                        )
                    )
            else:
            #  if len(self.returns) >= 3 else 0.0
                metrics['last_return_z'] = 0.0
        else:
            metrics['last_return_z'] = 0.0

        # Price Z-score relative to rolling price_window
        prices = np.array(
            self.portfolio_values[
                -(self.seq_len+1):
                # -(self.price_window+1):
                ], dtype=float
            )
        if prices.size > self.seq_len:
            past = prices[:-1]
            mu = past.mean()
            if past.size > 0:
                sigma = past.std() if past.std() > 1e-8 else 0.
            else:
                sigma = 0.0
            metrics['price_zscore'] = float(
                (prices[-1] - mu) / sigma
                ) if sigma > 0 else 0.0
        else:
            metrics['price_zscore'] = 0.0


        # Compute MFE and MAE over the window
        if r.size > 0:
            cumrets = np.cumprod(1 + r) - 1
            metrics['mfe'] = float(np.max(cumrets))
            metrics['mae'] = float(np.min(cumrets))
            # Tail risk: variance and downside volatility of the worst q%
            # var_lvl = np.percentile(r, 100 * self.var_percentile)
            var_lvl = np.percentile(r, 100 * 0.05)
            tail = r[r <= var_lvl]
            metrics['tail_variance'] = float(
                np.var(tail)) if tail.size > 0 and np.var(tail) > 1e-8 else 0.0
            metrics['downside_vol'] = float(
                np.std(tail)) if tail.size > 0 and np.std(tail) > 1e-8 else 0.0
        else:
            metrics['mfe'] = 0.0
            metrics['mae'] = 0.0
            metrics['tail_variance'] = 0.0
            metrics['downside_vol'] = 0.0

        # Compute price momentum z-score
        mh = np.array(self.momentum_history, dtype=float)
        if mh.size > 1:
            base = mh[:-1]
            mean_m = np.mean(base)
            if base.size > 0:
                std_m = np.std(base) if np.std(base) > 1e-8 else 0
            else:
                std_m = 0.0
            metrics['momentum_zscore'] = float(
                (mh[-1] - mean_m) / std_m
                ) if std_m > 1e-8 else 0.0
        else:
            metrics['momentum_zscore'] = 0.0

        # Return distribution entropy
        if r.size > 1:
            hist, _ = np.histogram(r, bins=self.entropy_bins, density=True)
            probs = hist / np.sum(hist + 1e-8)
            probs = probs[probs > 0]
            metrics['return_entropy'] = -float(np.sum(probs * np.log(probs)))
        else:
            metrics['return_entropy'] = 0.0

        # Variance ratio: var of k-step aggregated returns / var of single-step returns
        k = self.var_ratio_lag
        if r.size > k:
            if r.size > 0:
                agg = np.array([np.sum(r[i:i+k]) for i in range(len(r) - k + 1)])
                var_single = float(np.var(r)) if np.var(r) > 1e-8 else 0.
                metrics['variance_ratio'] = float(np.var(agg) / var_single) if var_single > 0. else 0.
            else:
                metrics['variance_ratio'] = 0.0
        else:
            metrics['variance_ratio'] = 0.0

        # Current win and loss streaks
        win_streak = 0
        for ret in r[::-1]:
            if ret > 0:
                win_streak += 1
            else:
                break
        loss_streak = 0
        for ret in r[::-1]:
            if ret < 0:
                loss_streak += 1
            else:
                break
        # metrics['win_streak'] = float(win_streak)
        # metrics['loss_streak'] = float(loss_streak)

        w = np.array(self.weights_history[-1], dtype=float)

        # 1. Effective number of bets (ENB)
        metrics['effective_bets'] = float(
            1.0 / np.sum(w**2)) if w.size > 0 else 0.0

        # 2. Mean absolute deviation (MAD)
        if r.size > 0:
            mean_r = np.mean(r)
            metrics['mad'] = float(
                np.mean(np.abs(r - mean_r)))
        else:
            metrics['mad'] = 0.0

        # 3. Gini coefficient of returns (shifted to positive)
        if r.size > 0:
            r_shift = r - np.min(r) + 1e-8
            sorted_r = np.sort(r_shift)
            n = sorted_r.size
            index = np.arange(1, n+1)
            gini = (
                2 * np.sum(index * sorted_r) / (n * np.sum(sorted_r))
                ) - ((n+1) / n)
            metrics['gini'] = float(gini)
        else:
            metrics['gini'] = 0.0

        # 4. Hill estimator for tail index (negative returns tail)
        '''if r.size > 1:
            var_lvl = np.percentile(r, 100 * 0.05) # self.var_percentile)
            tail = -r[r < var_lvl]  # convert to positive losses
            if tail.size > 1:
                sorted_tail = np.sort(tail)[::-1]
                k = sorted_tail.size
                x_k = sorted_tail[-1]
                # print(f'||| x_k: {x_k} ||| sorted_tail: {sorted_tail} ||| k: {k} ||| tail: {tail} ||| var_lvl: {var_lvl} ||| r: {r} |||')
                hill = float(np.mean(np.log(sorted_tail / x_k))) # if x_k > 0. and all(sorted_tail) > 0. else float(np.mean(np.log(np.abs(sorted_tail) / np.abs(x_k))))
                metrics['hill_estimator'] = hill
            else:
                metrics['hill_estimator'] = 0.0
        else:
            metrics['hill_estimator'] = 0.0''' #             self.historical_trades.at[self.timestamp, f'portfolio_{asset}_owned'] = float(self.portfolio.get(f'{asset}', 0.0))

            # Assign close quantities
            # self.historical_trades.at[self.timestamp, 'hrp_{asset}_quantity'] = float(float(self.hrp_portfolio.get(f'{asset}', 0.0)) - float(self.portfolio.get(f'{asset}', 0.0)))


        # 5. Drawdown at Risk (DaR)
        if dd.size > 0:
            metrics['drawdown_at_risk'] = float(
                np.percentile(
                    dd, 100 * (1 - 0.05) # self.var_percentile)
                    )
                )
        else:
            metrics['drawdown_at_risk'] = 0.0

        # 6. Coefficient of Variation (CoV)
        if r.size > 0:
            mean_r = np.mean(r)
            std_r = np.std(r) if np.std(r) > 1e-8 else 0.
            metrics['cov'] = float(std_r / abs(mean_r)) if mean_r != 0 else 0.0
        else:
            metrics['cov'] = 0.0

        # Latest values for env-specific features
        metrics['cash_ratio'] = self.cash_ratios[-1] if self.cash_ratios else 0.0
        metrics['asset_count'] = self.asset_counts[-1] if self.asset_counts else 0.0
        metrics['concentration_ratio'] = self.concentration_ratios[-1] if self.concentration_ratios else 0.0
        metrics['trade_count'] = sum(self.trade_counts)  # total trades in window
        metrics['portfolio_value_zscore'] = self.value_zscores[-1] if self.value_zscores else 0.0
        
        # Average win and loss returns
        wins = r[r > 0]
        losses = -r[r < 0]
        metrics['avg_win_return'] = float(wins.mean()) if wins.size > 0 else 0.0
        metrics['avg_loss_return'] = float(losses.mean()) if losses.size > 0 else 0.0

        # Max single-step drawdown in window
        pv = np.array(self.portfolio_values[-(self.seq_len+1):], dtype=float)
        if pv.size > 0:
            single_dd = (pv[:-1] - pv[1:]) / (pv[:-1] + 1e-8)
            metrics['max_single_drawdown'] = float(np.max(single_dd)) if single_dd.size > 0 else 0.0
        else:
            metrics['max_single_drawdown'] = 0.0

        # Avg trade size: turnover per trade
        trade_counts = np.array(self.trade_counts if hasattr(self, 'trade_counts') else [], dtype=np.float32)
        turnover_vals = np.array(self.concentration_ratios, dtype=float)  # reuse concentration buffer temporarily

        # Actually, trade_counts and turnover are in parent metrics, so compute here properly:
        # turnover = metrics.get('trade_count', 0.0)  # number of trades
        # total_turnover = metrics.get('turnover', 0.0)  # sum of weight changes
        total_trade_count = np.sum(trade_counts)
        metrics['avg_trade_size'] = float(turnover / total_trade_count) if total_trade_count > 0 else 0.0

        # Realized PnL metrics
        rp = np.array(self.realized_pnls, dtype=float)
        metrics['realized_pnl'] = float(rp[-1]) if rp.size > 0 else 0.0
        cum_return = float(self.cumulative_returns[-1]) if self.cumulative_returns else 0.0
        metrics['cum_return'] = cum_return

        # Episode progress
        # if self.horizon:
        #     step = len(self.portfolio_values) - 1
        #     metrics['episode_progress'] = float(step / self.horizon)
        # else:
        #     metrics['episode_progress'] = 0.0

        # Omega ratio: gains/losses at threshold = risk_free_rate
        # gains = r[r > self.risk_free_rate_] - self.risk_free_rate_
        # losses = self.risk_free_rate_ - r[r < self.risk_free_rate_]
        # metrics['omega'] = float(np.sum(gains) / np.sum(losses)) if losses.size > 0 else float('inf')

        # Sterling ratio: annualized return / avg max drawdown (approx)
        # ann_return = np.mean(r) * 252 if r.size > 0 else 0.0
        # avg_dd = np.mean(dd) if dd.size > 0 else 0.0
        # metrics['sterling_ratio'] = float(ann_return / avg_dd) if avg_dd != 0 else 0.0

        # Ulcer ratio: ulcer_index / cumulative return
        # cum_ret = metrics.get('cum_return', 0.0)
        # ui = metrics.get('ulcer_index', 0.0)
        metrics['ulcer_ratio'] = float(ulcer_index / abs(cum_return)) if cum_return != 0 else 0.0

        # Conditional Drawdown at Risk (CDaR): average of worst q% drawdowns
        if dd.size > 0:
            # q = int(np.ceil(self.var_percentile * dd.size))
            q = int(np.ceil(0.05 * dd.size))
            worst_dd = np.sort(dd)[-q:]
            metrics['cdar'] = float(np.mean(worst_dd))
        else:
            metrics['cdar'] = 0.0

        # Average holding period: window / number of trades
        # trade_count = metrics.get('trade_count', 0.0)
        trade_count = sum(self.trade_counts)
        # metrics['avg_holding_period'] = float(self.seq_len / trade_count) if trade_count != 0 else float(self.seq_len)

        # New high count: times current value equals new rolling max within window
        pv = np.array(self.portfolio_values[-(self.seq_len+1):], dtype=float)
        new_highs = np.sum(pv[1:] >= np.maximum.accumulate(pv[:-1]))
        # metrics['new_high_count'] = float(new_highs)

        # Recovery ratio: cumulative return / drawdown recovery time
        # rec_time = metrics.get('drawdown_recovery_time', 0.0)
        rec_time = recovery
        metrics['recovery_ratio'] = float(cum_return / rec_time) if rec_time > 0 else 0.0

        # Skewness-to-volatility ratio
        # vol = metrics.get('realized_vol', 0.0)
        # skew = metrics.get('skewness', 0.0)
        metrics['skew_vol_ratio'] = float(skewness / realized_vol) if realized_vol > 0 else 0.0

        n = r.size

        # Mean return and variance
        mean_r = float(np.mean(r)) if n > 0 else 0.0
        var_r = float(np.var(r)) if n > 0 and np.var(r) > 1e-8 else 0.0
        metrics['mean_return'] = mean_r
        metrics['variance'] = var_r

        # Range and IQR of returns
        if n > 0:
            metrics['range_return'] = float(np.max(r) - np.min(r))
            metrics['iqr_return'] = float(
                np.percentile(r, 75) - np.percentile(r, 25)
                )
        else:
            metrics['range_return'] = 0.0
            metrics['iqr_return'] = 0.0

        # Jarque-Bera statistic for normality
        S = skewness # metrics.get('skewness', 0.0)
        K = kurtosis # metrics.get('kurtosis', 0.0)
        metrics['jarque_bera'] = float(
            n * (
                S**2 / 6.0 + K**2 / 24.0
                )
            ) if n > 0 else 0.0

        # Autocorrelation of squared returns (lag 1)
        if n > 1 and self.current_step - self.start_step > 3:
            r2 = r**2
            # print(f'r: {r}')
            # autocorr_sq = float(np.corrcoef(r2[:-1], r2[1:])[0, 1])


            try:
                autocorr_sq = float(np.corrcoef(r2[:-1], r2[1:])[0, 1])
            except RuntimeWarning as e:
                print("Caught invalid‐divide warning:", e)
                # fallback to safe routine
                autocorr_sq = float(0.)



            if not np.isnan(autocorr_sq) and not np.isinf(autocorr_sq):
                metrics['autocorr_sq'] = autocorr_sq # float(
                    # np.corrcoef(
                    #     r2[:-1], r2[1:]
                    #     )[0, 1]
                    # )
                # print(f'autocorr_sq : {autocorr_sq}')
            else:
                metrics['autocorr_sq'] = 0.0
        else:
            metrics['autocorr_sq'] = 0.0

        # Drawdown rate and entropy
        m = dd.size
        if m > 0:
            metrics['drawdown_rate'] = float(np.sum(dd > 0) / m)
            # Entropy of drawdown distribution
            hist, _ = np.histogram(
                dd, bins=self.entropy_bins, density=True
                )
            probs = hist / np.sum(hist + 1e-8)
            probs = probs[probs > 0]
            metrics['drawdown_entropy'] = float(
                -np.sum(
                    probs * np.log(probs)
                    )
                )
        else:
            metrics['drawdown_rate'] = 0.0
            metrics['drawdown_entropy'] = 0.0

        # Max drawdown length ratio
        # # max_dd_len = metrics.get('drawdown_length', 0.0)
        # max_dd_len = self._max_drawdown_length
        # metrics['max_drawdown_length_ratio'] = (
        #     float(max_dd_len / self.seq_len) if self.seq_len > 0 else 0.0
        # )

        # Win/Loss ratio
        wins = np.sum(r > 0)
        losses = np.sum(r < 0)
        metrics['win_loss_ratio'] = float(
            wins / losses) if losses != 0 else float(wins)

        # Average win streak and loss streak
        def avg_streak(arr, condition):
            streaks = []
            count = 0
            for val in arr:
                if condition(val):
                    count += 1
                else:
                    if count > 0:
                        streaks.append(count)
                        count = 0
            if count > 0:
                streaks.append(count)
            return float(np.mean(streaks)) if streaks else 0.0

        metrics['avg_win_streak'] = avg_streak(r, lambda x: x > 0)
        metrics['avg_loss_streak'] = avg_streak(r, lambda x: x < 0)

        '''# Tail return ratio: 95th percentile / abs(5th percentile)
        if r.size > 0:
            p95 = np.percentile(r, 95)
            p05 = np.percentile(r, 5)
            metrics['tail_return_ratio'] = float(p95 / abs(p05)) if p05 != 0 else 0 # float('inf')
        else:
            metrics['tail_return_ratio'] = 0.0'''

        # Spectral entropy of returns
        if r.size > 1:
            fft_vals = np.fft.fft(r)
            psd = np.abs(fft_vals)**2
            psd = psd[:len(psd)//2]  # keep positive freqs
            psd_sum = np.sum(psd)
            if psd_sum > 0:
                psd_norm = psd / psd_sum
                psd_norm = psd_norm[psd_norm > 0]
                metrics['spectral_entropy'] = -float(np.sum(psd_norm * np.log(psd_norm)))
            else:
                metrics['spectral_entropy'] = 0.0
        else:
            metrics['spectral_entropy'] = 0.0

        # Average drawdown magnitude when in drawdown
        if dd.size > 0:
            dd_positive = dd[dd > 0]
            metrics['avg_drawdown'] = float(np.mean(dd_positive)) if dd_positive.size > 0 else 0.0
        else:
            metrics['avg_drawdown'] = 0.0


        # Annualized metrics (assume daily data, 252 trading days)
        metrics['annual_return'] = float(np.mean(r) * 252) if n > 1 else 0.0
        metrics['annual_volatility'] = float(np.std(r) * np.sqrt(252)) if n > 1 and np.std(r) > 1e-8 else 0.0
        if metrics['annual_volatility'] > 0:
            metrics['annual_sharpe'] = metrics['annual_return'] / metrics['annual_volatility']
        else:
            metrics['annual_sharpe'] = 0.0

        # CAGR: compound annual growth rate
        if len(self.portfolio_values) > 1:
            init = self.portfolio_values[0]
            curr = self.portfolio_values[-1]
            years = (
                len(self.portfolio_values) - 1
                ) / 252
            if init > 0 and years > 0:
                metrics['cagr'] = float(
                    (curr / init) ** (1 / years) - 1
                    )
            else:
                metrics['cagr'] = 0.0
        else:
            metrics['cagr'] = 0.0

        # Trend slope & R-squared of portfolio values
        if pv.size > 1:
            x = np.arange(pv.size)
            slope, intercept = np.polyfit(x, pv, 1)
            yhat = slope * x + intercept
            ss_res = np.sum(
                (pv - yhat) ** 2
                )
            ss_tot = np.sum(
                (pv - np.mean(pv)) ** 2
                )
            metrics['trend_slope'] = float(slope)
            metrics['trend_r2'] = float(
                1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            metrics['trend_slope'] = 0.0
            metrics['trend_r2'] = 0.0

        # Return consistency: proportion of returns within ±1 std
        if n > 1:
            mean_r = np.mean(r)
            std_r = np.std(r) if np.std(r) > 1e-8 else 0.
            within = np.sum(np.abs(r - mean_r) <= std_r)
            metrics['return_consistency'] = float(within / n)
        else:
            metrics['return_consistency'] = 0.0

        '''# Zero-crossing rate of returns
        if n > 1:
            signs = np.sign(r)
            crossings = np.sum(signs[:-1] != signs[1:])
            metrics['zero_crossing_rate'] = float(crossings / (n - 1))
        else:
            metrics['zero_crossing_rate'] = 0.0'''


        return metrics

        # if self.feature_list is None:
        #     names = sorted(metrics.keys())
        # else:
        #     names = self.feature_list
        # return np.array([metrics[name] for name in names], dtype=np.float32)

    def init_historical_trades(self, debug=False):
        # self.data.set_index('timestamp', inplace=True)
        self.historical_trades = pd.DataFrame(
            # self.data.loc[self.start_step:self.end_step]['timestamp']
            self.data.loc[:self.end_step+1]['timestamp']
            )
        self.historical_trades['timestamp'] = pd.to_datetime(
            self.historical_trades['timestamp'], unit='ns' # 'd'
            )
        self.historical_trades.set_index('timestamp', inplace=True)

        self.historical_trades['cash_balance'] = float(0.)
        self.historical_trades['previous_cash_balance'] = float(0.)
        self.historical_trades['portfolio_value'] = float(0.)
        self.historical_trades['hrp_portfolio_value'] = float(0.)


        self.historical_trades['portfolio_value'] = float(self.initial_balance)
        self.historical_trades['hrp_portfolio_value'] = float(self.initial_balance)
        self.historical_trades['portfolio_value_miuns_hrp_portfolio_value'] = float(0.)
        self.historical_trades['var'] = float(0.)
        self.historical_trades['max_drawdown'] = float(0.)
        self.historical_trades['parametric_var'] = float(0.)
        self.historical_trades['parametric_var_dollar'] = float(0.)
        self.historical_trades['prev_parametric_var_dollar'] = float(0.)

        self.historical_trades['cvar'] = float(0.)
        self.historical_trades['parametric_cvar'] = float(0.)
        self.historical_trades['cvar_dollar'] = float(0.)
        self.historical_trades['previous_profit'] = float(0.)
        self.historical_trades['profit'] = float(0.)
        self.historical_trades['prev_var'] = float(0.)
        self.historical_trades['prev_parametric_var'] = float(0.)
        self.historical_trades['prev_parametric_cvar'] = float(0.)
        self.historical_trades['prev_cvar_dollar'] = float(0.)

        self.historical_trades['return_'] = float(0.)
        self.historical_trades['prev_return_'] = float(0.)


        self.historical_trades[f'portfolio_sold_value'] = float(0.)
        self.historical_trades[f'portfolio_bought_value'] = float(0.)
        self.historical_trades[f'total_transaction_fee'] = float(0.)
        self.historical_trades[f'portfolio_total_transactions_value'] = float(0.)
        self.historical_trades['before_buy_cash'] = float(self.balance)

        self.historical_trades[f"pf_alpha_monthly"] = float(0.)
        self.historical_trades[f"pf_vol_monthly"] = float(0.)

        self.historical_trades[f"pf_sharpe"] = float(0.)
        self.historical_trades[f"pf_sortino"] = float(0.)
        self.historical_trades[f"pf_max_drawdown"] = float(0.)
        self.historical_trades[f"pf_beta"] = float(0.)
        self.historical_trades[f"pf_beta_adj_sharpe"] = float(0.)
        self.historical_trades["pf_m_squared_ratio"] = float(0.)
        self.historical_trades[f'should_be_portfolio_values'] = float(0.)

        self.historical_trades[f'Portfolio_Dirchlet_Posterior_entropy_getting_blocked'] = float(0.)


        current_weights = self.get_current_weights # [1:] # If cash, self.get_current_weights[0], doesn't count
        previous_weights = self.previous_weights
        turn_over_rates = [np.abs(a - b) for a, b in zip(current_weights, previous_weights)] #  turnover_accum += 0.5 * np.abs(action - prev_weights).sum()

        for asset, i in zip(self.selected_assets, range(len(self.actions))):

            self.historical_trades[f'{asset}_Dirchlet_Posterior_mu_getting_blocked'] = float(0.)
            self.historical_trades[f'{asset}_Dirchlet_Posterior_variance_getting_blocked'] = float(0.)


            self.historical_trades[f'hrp_{asset}_action'] = float(0.)
            self.historical_trades[f'{asset}_buy_hold_minus_portfolio'] = float(0.)
            self.historical_trades[f'hrp_{asset}_buy_hold_minus_portfolio'] = float(0.)

            self.historical_trades[f"FIFO_realized_{asset}_PnL"] = float(0.)
            self.historical_trades[f"FIFO_unrealized_{asset}_PnL"] = float(0.)
            self.historical_trades[f"LIFO_realized_{asset}_PnL"] = float(0.)
            self.historical_trades[f"LIFO_unrealized_{asset}_PnL"] = float(0.)
            self.historical_trades[f"{asset}_FIFO_Total_PnL"] = float(0.)
            self.historical_trades[f"{asset}_LIFO_Total_PnL"] = float(0.)
            


            self.historical_trades[f"{asset}_PnL"] = float(0.)
            self.historical_trades[f'portfolio_{asset}_owned'] = float(0.) # self.portfolio[asset]
            self.historical_trades[f'hrp_{asset}_quantity'] = float(0.)
            self.historical_trades[f'hrp_portfolio_{asset}_owned'] = float(0.) # self.hrp_portfolio[asset]


            self.historical_trades[f'{asset}_trade_amount'] = float(0.)
            self.historical_trades[f'{asset}_cash_cost_revenue'] = float(0.)

            self.historical_trades[f'hrp_{asset}_trade_amount'] = float(0.)
            self.historical_trades[f'hrp_{asset}_cash_cost_revenue'] = float(0.)

            self.historical_trades[f'{asset}_transaction_fee'] = float(0.)
            self.historical_trades[f'{asset}_trade_value'] = float(0.)

        self.historical_trades["portfolio_turn_over_rate"] = float(np.sum(turn_over_rates))

        if not self.use_action_norm:
            self.historical_trades['cash_trade_action'] = float(0.)
            self.historical_trades['cash_intended_trade_weight'] = float(0.)

        for asset in self.portfolio.keys():
            self.historical_trades[f'{asset}_turn_over_rate'] = float(turn_over_rates[i])
            self.historical_trades[f'{asset}_previous_values'] = float(0.) # previous_asset_value
            self.historical_trades[f'{asset}_delta_values'] = float(0.) # float(asset_value - previous_asset_value) / previous_asset_value) if previous_asset_value != 0. else 0.
            # self.btc_quantity
            if self.use_action_norm:
                self.historical_trades[f'{asset}_trade_action'] = float(0.)
                self.historical_trades[f'{asset}_intended_trade_weight'] = float(0.) # self.action_weights
            self.historical_trades[f'{asset}_values'] = float(0.)
            self.historical_trades[f'{asset}_weights'] = float(0.) # float(current_weights[i])  # float(0.)
            self.historical_trades[f'{asset}_previous_weights'] = float(0.) # float(previous_weights[i]) # float(0.)
            self.historical_trades[f'{asset}_delta_weights'] = float(0.) # float(float(current_weights[i]) - float(previous_weights[i]))
            self.historical_trades[f'hrp_{asset}_values'] = float(0.)
            self.historical_trades[f'hrp_{asset}_weights'] = float(0.)
            self.historical_trades[f'portfolio_miuns_{asset}_values'] = float(0.)
            self.historical_trades[f'hrp_portfolio_miuns_{asset}_values'] = float(0.)

        self.historical_trades[self.portfolio_metrics_col] = float(0.)

        # Ensure the price data has a datetime index.
        if not isinstance(self.price_data.index, pd.DatetimeIndex):
            if 'timestamp' in self.price_data.columns:
                self.price_data.set_index('timestamp', inplace=True)
                logging.info(
                    "Price data index set to timestamp."
                    )
            else:
                logging.error(
                    "Price data does not contain a 'timestamp' column."
                    )
                raise ValueError(
                    "Missing 'timestamp' column in price_data"
                    )
                
        # Historical trades is expected to be a DataFrame where we will record buy-hold values.
        # If the historical trades do not have a datetime index, try to set one using the 'timestamp' column.
        if not isinstance(self.historical_trades.index, pd.DatetimeIndex):
            if 'timestamp' in self.historical_trades.columns:
                self.historical_trades.set_index('timestamp', inplace=True)
                logging.info(
                    "Historical trades index set to timestamp column."
                    )
            else:
                logging.warning(
                    "Historical trades do not have a timestamp column; index alignment may fail."
                    )

        self.init_buy_holds(index=self.start_step)

        # # Initialize the live PnL calculator.
        # self.live_pnl = LivePnLCalculator(self.selected_assets)
        # # Initialize dictionary to hold the previous total PnL per asset for trade evaluation.
        # self.last_asset_pnl = {asset: 0.0 for asset in self.selected_assets}
        # self.method = 'FIFO' # 'LIFO'
        self.live_pnl.reset()
        current_weights = np.array(self.get_current_weights)
        if debug:
            print(
                f'||| current_weights: {current_weights} ||| current_weights.shape: {current_weights.shape}'
                )
            
        self.update(
            portfolio_value=float(
                self.portfolio_value
                ),
            weights=current_weights,
            benchmark_return=self.benchmark_return
            )


        self.hrp_portfolio_values.append(self.hrp_portfolio_value)

        # metrics = self.compute_metrics()
        if self.current_step - self.start_step > 0:
            metrics = self.compute_metrics()
        else:
            metrics = self.metrics
        if debug:
            print(f'||| metrics: {metrics} |||')
            self.debug_missing_columns(metrics)
        else:
            self.record_historical_data(metrics)

        # self.record_historical_data() # Original.
        if self.initialized == False:
            self.running_scaler.fit(self.historical_trades) # .replace([-np.inf, np.inf], np.nan).fillna(value=0))
            # col = [self.historical_trades.columns]

            self.initialized = True









    def init_env_attributes(self, debug=False):
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ns') # 'd') # 's')

        self.r_ = float(0.)
        self.prev_r_ = float(0.)
        self.prospect_theory_loss_aversion_return = float(0.)
        self.prev_prospect_theory_loss_aversion_return = float(0.)
        self.traded = 0.
        self.should_be_portfolio_value = float(0.)

        self._peak_value = None
        self._peak_index = None
        self._max_drawdown = 0.0
        self._max_drawdown_length = 0
        self._current_drawdown_start = None
        self._current_drawdown_start: Optional[int] = None

        
        self.weights_history: List[np.ndarray] = []
        self.benchmark_returns: List[float] = []
        self.drawdowns: List[float] = []
        self.vol_history: List[float] = []
        self.portfolio_values = []
        self.momentum_history : List[float] = []


        self.tsne_embeddings = None  # Placeholder for t-SNE embeddings
        self.warmup_threshold = 1000  # Number of observations needed to fit PCA

        self.rewards_sum = 0
        self.previous_hrp_portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance # self.portfolio_values[-1] if len(self.portfolio_values) > 0 else float(self.portfolio_value)


        self.raw_actions = [] # 0, 0, 0]
        self.raw_hrp_actions = [] # 0, 0, 0]
        self.actions = [] # 0, 0, 0] # self.action_space.sample() # [0, 0, 0] # Hard coded as of now. In range of len(self,selected_assets).
        self.hrp_actions = [] # 0, 0, 0]
        self.trade_actions = []
        self.action_weights = []
        self.current_weights.append(1.)
        self.previous_weights.append(1.)
        self.sum_trade_actions = float(0.0)

        for i in range(len(self.selected_assets)):
            self.raw_actions.append(float(0.))
            self.raw_hrp_actions.append(float(0.))
            self.actions.append(float(0.)) # self.action_space.sample() # [0, 0, 0] # Hard coded as of now. In range of len(self,selected_assets).
            self.hrp_actions.append(float(0.))
            self.trade_actions.append(float(0.))
            self.action_weights.append(float(0.))
            self.current_weights.append(float(0.))
            self.previous_weights.append(float(0.))

        self.is_today_month_end = False
        self.is_successful_month_end = False

        self.is_traded = False
        self.prev_balance = self.initial_balance
        self.prev_prev_balance = self.prev_balance
        self.hrp_prev_balance = self.initial_balance
        self.current_step = self.start_step
        self.previous_step = self.start_step
        self.check_returns = 0  # Resetting to an empty list
        self.reward = 0
        

        self.action = 0
        self.rewards = []
        self.check_returns = 0
        self.price = 0
        
        self.cash_trade_action = float(0.)
        self.cash_intended_trade_weight = float(0.)

        self.hrp_btc_quantity = 0
        self.hrp_btc_price = 0
        self.hrp_eth_price = 0
        self.hrp_sp500_price = 0
        self.hrp_action = 0
        self.hrp_eth_quantity = 0
        self.hrp_sp500_quantity = 0
        self.hrp_btc_quantity = 0

        self.var = float(0.)
        self.max_drawdown = float(0.)
        self.parametric_var = float(0.)
        self.cvar = float(0.)
        self.parametric_cvar = float(0.)
        self.cvar_dollar = float(0.)
        self.previous_profit = float(0.)
        self.profits = []
        self.profit = float(0.)
        self.previous_cash = float(self.portfolio['cash'])

        self.prev_max_drawdown = float(0.)
        self.prev_var = float(0.)
        self.prev_parametric_var = float(0.)
        self.prev_cvar = float(0.)
        self.prev_parametric_cvar = float(0.)
        self.prev_cvar_dollar = float(0.)

        self.return_ = float(0.)
        self.prev_return_ = float(0.)
        self.parametric_var_dollar = float(0.)
        self.prev_parametric_var_dollar = float(0.)



        self.hrp_portfolio_values = []
        self.historical_pct_returns = []

        self.portfolio_sold_value = float(0.)
        # self.portfolio_bought_value = float(0.)
        self.portfolio_bought_value = float(0.)
        self.total_transaction_fee = float(0.)
        self.portfolio_total_transactions_value = float(0.)

        self.fixed_fee = 1.0             # Flat fee per trade (e.g., $1)
        self.proportional_fee = 0.002    # Proportional fee (e.g., 0.1%)
        self.min_trade_value = 10.0      # Minimum notional trade value to execute
        

        # self.portfolio_value = self.balance + self.price * self.btc_quantity
        self.portfolio = {'cash': self.initial_balance}
        # self.portfolio = {'cash': 1000}
        self.portfolio.update({asset: 0 for asset in self.selected_assets})

        self.hrp_portfolio = {'cash': self.initial_balance}
        self.hrp_portfolio.update({asset: 0 for asset in self.selected_assets})

        # self.previous_weights
        self.previous_portfolio = {'cash': self.initial_balance}
        # self.portfolio = {'cash': 1000}
        self.previous_portfolio.update({asset: 0 for asset in self.selected_assets})
        self.metrics.update({col: 0 for col in self.portfolio_metrics_col})
        # print(f'self.metrics : {self.metrics}')

        self.init_buy_holds(index=self.start_step)

        # # Initialize the live PnL calculator.
        # self.live_pnl = LivePnLCalculator(self.selected_assets)
        # # Initialize dictionary to hold the previous total PnL per asset for trade evaluation.
        # self.last_asset_pnl = {asset: 0.0 for asset in self.selected_assets}
        # self.method = 'FIFO' # 'LIFO'
        self.live_pnl.reset()
        current_weights = np.array(self.get_current_weights)
        if debug:
            print(
                f'||| current_weights: {current_weights} ||| current_weights.shape: {current_weights.shape}'
                )
            
        self.update(
            portfolio_value=float(
                self.portfolio_value
                ),
            weights=current_weights,
            benchmark_return=self.benchmark_return
            )

        self.hrp_portfolio_values.append(self.hrp_portfolio_value)

        # metrics = self.compute_metrics()
        if self.current_step - self.start_step > 0:
            metrics = self.compute_metrics()
        else:
            metrics = self.metrics
        if debug:
            print(f'||| metrics: {metrics} |||')
            self.debug_missing_columns(metrics)
        else:
            self.record_historical_data(metrics)






    def reset_env_attributs(self, debug=False):
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ns') # 'd') # 's')

        self.r_ = float(0.)
        self.prev_r_ = float(0.)
        self.prospect_theory_loss_aversion_return = float(0.)
        self.prev_prospect_theory_loss_aversion_return = float(0.)
        self.traded = 0.
        self.should_be_portfolio_value = float(0.)
        self.should_be_portfolio_values.clear()

        self._peak_value = None
        self._peak_index = None
        self._max_drawdown = 0.0
        self._max_drawdown_length = 0
        self._current_drawdown_start = None
        self._current_drawdown_start: Optional[int] = None

        self.weights_history.clear()
        self.benchmark_returns.clear()
        self.drawdowns.clear()
        self.vol_history.clear()
        self.portfolio_values.clear()
        self.momentum_history.clear()
        self.cash_ratios.clear() # : List[float] = []
        self.asset_counts.clear() # : List[float] = []
        self.concentration_ratios.clear() # : List[float] = []
        self.trade_counts.clear() # : List[float] = []
        self.value_zscores.clear()
        self.realized_pnls.clear()
        self.cumulative_returns.clear()



        self.tsne_embeddings = None  # Placeholder for t-SNE embeddings
        self.warmup_threshold = 1000  # Number of observations needed to fit PCA

        self.rewards_sum = 0
        self.previous_hrp_portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance # self.portfolio_values[-1] if len(self.portfolio_values) > 0 else float(self.portfolio_value)

        self.raw_actions.clear()
        self.raw_hrp_actions.clear()
        self.actions.clear()
        self.hrp_actions.clear()
        self.trade_actions.clear()
        self.action_weights.clear()
        self.current_weights.clear()

        self.previous_weights.clear()
        self.current_weights.append(1.)
        self.previous_weights.append(1.)
        self.sum_trade_actions = float(0.0)

        for i in range(len(self.selected_assets)):
            self.raw_actions.append(float(0.))
            self.raw_hrp_actions.append(float(0.))
            self.actions.append(float(0.)) # self.action_space.sample() # [0, 0, 0] # Hard coded as of now. In range of len(self,selected_assets).
            self.hrp_actions.append(float(0.))
            self.trade_actions.append(float(0.))
            self.action_weights.append(float(0.))
            self.current_weights.append(float(0.))
            self.previous_weights.append(float(0.))

        self.is_today_month_end = False
        self.is_successful_month_end = False

        self.is_traded = False
        # self.balance = self.initial_balance
        self.prev_balance = self.initial_balance
        self.prev_prev_balance = self.prev_balance
        self.hrp_prev_balance = self.initial_balance
        self.current_step = self.start_step
        self.previous_step = self.start_step
        self.check_returns = 0  # Resetting to an empty list
        self.reward = 0
        

        self.action = 0
        self.returns.clear() # = []
        self.rewards.clear()
        self.check_returns = 0
        self.price = 0
        
        self.cash_trade_action = float(0.)
        self.cash_intended_trade_weight = float(0.)

        self.hrp_btc_quantity = 0
        self.hrp_btc_price = 0
        self.hrp_eth_price = 0
        self.hrp_sp500_price = 0
        self.hrp_action = 0
        self.hrp_eth_quantity = 0
        self.hrp_sp500_quantity = 0
        self.hrp_btc_quantity = 0

        self.var = float(0.)
        self.max_drawdown = float(0.)
        self.parametric_var = float(0.)
        self.cvar = float(0.)
        self.parametric_cvar = float(0.)
        self.cvar_dollar = float(0.)
        self.previous_profit = float(0.)
        self.profits.clear()
        self.profit = float(0.)
        self.previous_cash = float(self.portfolio['cash'])

        self.prev_max_drawdown = float(0.)
        self.prev_var = float(0.)
        self.prev_parametric_var = float(0.)
        self.prev_cvar = float(0.)
        self.prev_parametric_cvar = float(0.)
        self.prev_cvar_dollar = float(0.)
        self.return_ = float(0.)
        self.prev_return_ = float(0.)
        self.parametric_var_dollar = float(0.)
        self.prev_parametric_var_dollar = float(0.)

        self.hrp_portfolio_values.clear()
        self.historical_pct_returns.clear()

        self.portfolio_sold_value = float(0.)
        self.portfolio_bought_value = float(0.)
        self.total_transaction_fee = float(0.)
        self.portfolio_total_transactions_value = float(0.)

        self.fixed_fee = 1.0             # Flat fee per trade (e.g., $1)
        self.proportional_fee = 0.002    # Proportional fee (e.g., 0.1%)
        self.min_trade_value = 10.0      # Minimum notional trade value to execute
        

        # self.portfolio_value = self.balance + self.price * self.btc_quantity
        self.portfolio = {'cash': self.initial_balance}
        # self.portfolio = {'cash': 1000}
        self.portfolio.update({asset: 0 for asset in self.selected_assets})

        self.hrp_portfolio = {'cash': self.initial_balance}
        self.hrp_portfolio.update({asset: 0 for asset in self.selected_assets})

        # self.previous_weights
        self.previous_portfolio = {'cash': self.initial_balance}
        # self.portfolio = {'cash': 1000}
        self.previous_portfolio.update({asset: 0 for asset in self.selected_assets})
        self.metrics.update({col: 0 for col in self.portfolio_metrics_col})
        # print(f'self.metrics : {self.metrics}')

        self.init_buy_holds(index=self.start_step)

        # # Initialize the live PnL calculator.
        # self.live_pnl = LivePnLCalculator(self.selected_assets)
        # # Initialize dictionary to hold the previous total PnL per asset for trade evaluation.
        # self.last_asset_pnl = {asset: 0.0 for asset in self.selected_assets}
        # self.method = 'FIFO' # 'LIFO'
        self.live_pnl.reset()
        current_weights = np.array(self.get_current_weights)

        if debug:
            print(
                f'||| current_weights: {current_weights} ||| current_weights.shape: {current_weights.shape}'
                )
            
        self.update(
            portfolio_value=float(
                self.portfolio_value
                ),
            weights=current_weights,
            benchmark_return=self.benchmark_return
            )

        self.hrp_portfolio_values.append(self.hrp_portfolio_value)

        # metrics = self.compute_metrics()
        if self.current_step - self.start_step > 0:
            metrics = self.compute_metrics()
        else:
            metrics = self.metrics
        if debug:
            print(f'||| metrics: {metrics} |||')
            self.debug_missing_columns(metrics)
        else:
            self.record_historical_data(metrics)




    def reset(self, debug=False):
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ns') # 'd') # 's')

        self.r_ = float(0.)
        self.prev_r_ = float(0.)
        self.prospect_theory_loss_aversion_return = float(0.)
        self.prev_prospect_theory_loss_aversion_return = float(0.)
        self.traded = 0.
        self.should_be_portfolio_value = float(0.)
        self.should_be_portfolio_values.clear()

        self.hrp_previous_weights.clear()


        self._peak_value = None
        self._peak_index = None
        self._max_drawdown = 0.0
        self._max_drawdown_length = 0
        self._current_drawdown_start = None
        self._current_drawdown_start: Optional[int] = None

        self.weights_history: List[np.ndarray] = []
        self.benchmark_returns: List[float] = []
        self.drawdowns: List[float] = []
        self.vol_history: List[float] = []
        self.portfolio_values = []
        self.momentum_history : List[float] = []
        self.cash_ratios.clear() # : List[float] = []
        self.asset_counts.clear() # : List[float] = []
        self.concentration_ratios.clear() # : List[float] = []
        self.trade_counts.clear() # : List[float] = []
        self.value_zscores.clear()
        self.realized_pnls.clear()
        self.cumulative_returns.clear()

        self.rewards_sum = 0
        self.previous_hrp_portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance # self.portfolio_values[-1] if len(self.portfolio_values) > 0 else float(self.portfolio_value)

        self.raw_actions = [] # 0, 0, 0]
        self.raw_hrp_actions = [] # 0, 0, 0]
        self.actions = [] # 0, 0, 0] # self.action_space.sample() # [0, 0, 0] # Hard coded as of now. In range of len(self,selected_assets).
        self.hrp_actions = [] # 0, 0, 0]
        self.trade_actions = []
        self.action_weights = []
        self.current_weights.clear()
        self.previous_weights.clear()
        self.current_weights.append(1.)
        self.previous_weights.append(1.)
        self.sum_trade_actions = float(0.0)

        for i in range(len(self.selected_assets)):
            self.raw_actions.append(float(0.))
            self.raw_hrp_actions.append(float(0.))
            self.actions.append(float(0.)) # self.action_space.sample() # [0, 0, 0] # Hard coded as of now. In range of len(self,selected_assets).
            self.hrp_actions.append(float(0.))
            self.trade_actions.append(float(0.))
            self.action_weights.append(float(0.))
            self.current_weights.append(float(0.))
            self.previous_weights.append(float(0.))

        self.is_today_month_end = False
        self.is_successful_month_end = False

        self.is_traded = False
        self.prev_balance = self.initial_balance
        self.prev_prev_balance = self.prev_balance
        self.hrp_prev_balance = self.initial_balance
        self.current_step = self.start_step
        self.previous_step = self.start_step
        self.check_returns = 0  # Resetting to an empty list
        self.reward = 0

        self.action = 0
        self.returns.clear() # = []
        self.rewards = []
        self.check_returns = 0
        self.price = 0
        

        self.cash_trade_action = float(0.)
        self.cash_intended_trade_weight = float(0.)

        self.hrp_btc_quantity = 0
        self.hrp_btc_price = 0
        self.hrp_eth_price = 0
        self.hrp_sp500_price = 0
        self.hrp_action = 0
        self.hrp_eth_quantity = 0
        self.hrp_sp500_quantity = 0
        self.hrp_btc_quantity = 0

        self.var = float(0.)
        self.max_drawdown = float(0.)
        self.parametric_var = float(0.)
        self.cvar = float(0.)
        self.parametric_cvar = float(0.)
        self.cvar_dollar = float(0.)
        # self.previous_profits = 
        self.previous_profit = float(0.)
        self.profits = []
        self.profit = float(0.)
        self.previous_cash = float(self.portfolio['cash'])

        self.prev_max_drawdown = float(0.)
        self.prev_var = float(0.)
        self.prev_parametric_var = float(0.)
        self.prev_cvar = float(0.)
        self.prev_parametric_cvar = float(0.)
        self.prev_cvar_dollar = float(0.)

        self.return_ = float(0.)
        self.prev_return_ = float(0.)
        self.parametric_var_dollar = float(0.)
        self.prev_parametric_var_dollar = float(0.)



        self.hrp_portfolio_values = []
        self.historical_pct_returns = []

        self.portfolio_sold_value = float(0.)
        self.portfolio_bought_value = float(0.)
        self.total_transaction_fee = float(0.)
        self.portfolio_total_transactions_value = float(0.)

        self.fixed_fee = 1.0             # Flat fee per trade (e.g., $1)
        self.proportional_fee = 0.002    # Proportional fee (e.g., 0.1%)
        self.min_trade_value = 10.0      # Minimum notional trade value to execute
        self.engine.reset()

        # Create a tracker for block‐probabilities
        self.block_tracker.reset()
        uniform_prior = np.ones(self.n_assets)/self.n_assets


        self.portfolio = {'cash': self.initial_balance}
        # self.portfolio = {'cash': 1000}
        self.portfolio.update({asset: 0 for asset in self.selected_assets})

        self.hrp_portfolio = {'cash': self.initial_balance}
        self.hrp_portfolio.update({asset: 0 for asset in self.selected_assets})

        # self.previous_weights
        self.previous_portfolio = {'cash': self.initial_balance}
        # self.portfolio = {'cash': 1000}
        self.previous_portfolio.update({asset: 0 for asset in self.selected_assets})
        self.metrics.update({col: 0 for col in self.portfolio_metrics_col})
        # print(f'self.metrics : {self.metrics}')

        self.cumulative_returns_history = []  # To track cumulative returns over time
        self.cumulative_history = []
        self.previous_returns = []
        
        portfolio_value = self.balance + self.btc_quantity * self.price
        self.portfolio_values.append(portfolio_value)
        portfolio_returns = self.cumulative_returns_history #self.portfolio_returns
        cumulative_return = (portfolio_value / self.initial_balance) - 1
        self.cumulative_returns_history.append(cumulative_return)

        self.max_step = self.end_step - 1 - self.start_step
        self.max_ep_len = int(self.end_step - self.start_step)


        self.reset_env_attributs()
        del self.historical_trades
        self.historical_trades = deepcopy(self.df_raw)
        for name, fe in self.fe_dict.items():
            fe.reset()


            current_weights = self.get_current_weights
            hrp_current_weights = self.hrp_get_current_weights[1:]
            hrp_previous_weights = self.hrp_get_current_weights[1:]
            current_prices = self.get_current_prices_dict()
            previous_prices = self.get_previous_prices_dict()
            next_prices = self.get_next_prices_dict()
            previous_weights = self.previous_weights
            names_list = []
            asset_weights = current_weights[1:]
            previous_asset_weights = self.previous_weights[1:]
            hrp_prev_asset_quantity_list = []
            hrp_current_asset_quantity_list = []

            for i in range(len(self.selected_assets)):
                asset_value = float(
                    self.portfolio_value * asset_weights[i]
                    )
                previous_asset_value = float(
                    self.previous_portfolio_value * previous_asset_weights[i]
                    )
                hrp_asset_value = float(
                    self.hrp_portfolio_value * hrp_current_weights[i]
                    )
                hrp_previous_asset_value = float(
                    self.previous_hrp_portfolio_value * hrp_previous_weights[i]
                    )
                

                asset = self.selected_assets[i] # Original.

                hrp_prev_asset_quantity = float(
                    float(
                        self.historical_trades.at[
                            # self.prev_previous_timestamp,
                            self.previous_timestamp,
                            f'hrp_portfolio_{asset}_owned'
                            ]
                        )
                    )
                hrp_current_asset_quantity = float(
                    float(
                        self.hrp_portfolio.get(f'{asset}', 0.0)

                        )
                    )
                hrp_should_be_asset_value = hrp_prev_asset_quantity * current_prices[asset] if asset != 'cash' else hrp_prev_asset_quantity
                hrp_current_asset_value = hrp_current_asset_quantity * current_prices[asset] if asset != 'cash' else hrp_current_asset_quantity
                hrp_asset_return = float(float(hrp_current_asset_value - hrp_should_be_asset_value) / self.previous_hrp_portfolio_value) if self.previous_hrp_portfolio_value != 0. else 0.
                fe.update(
                    asset_value=asset_value,
                    weight=asset_weights[i],
                    current_price=current_prices[asset] if asset != 'cash' else float(1.), # self.historical_trades.at[self.timestamp, 'cash_balance']
                    previous_price=previous_prices[asset] if asset != 'cash' else float(1.),
                    previous_weight=previous_asset_weights[i],
                    current_weight=asset_weights[i],
                    prev_asset_quantity=float(
                        self.historical_trades.at[
                            self.previous_timestamp, f'portfolio_{asset}_owned'
                            ]
                        ) if asset != 'cash' else float(self.prev_balance),
                    prev_balance=self.prev_balance,
                    cash=self.portfolio['cash'],
                    benchmark_return=hrp_asset_return,
                    debug=debug
                )



        if not self.use_seq_obs:
            obs = self.get_init_state()
        else:
            obs = self.get_seq_state(window_size=self.seq_len)
        return obs 



    def debug_missing_columns_original_df(self):
        col_list1 = self.portfolio_metrics_col
        col_list2 = [] # self.keys_list
        """for col1 in self.data.columns:
            col_list1.append(col1)"""
        logging.info(
f'''
||| Before recording historical data:
||| {col_list1}
||| len(col_list1):
||| {len(col_list1)}
'''
        )
        self.record_historical_data(self.metrics)
        for col2 in self.historical_trades.columns:
            col_list2.append(col2)
        logging.info(
f'''
||| After recording historical data:
||| {col_list2}
||| len(col_list2):
||| {len(col_list2)}
''' 
        )
        set1 = set(col_list1)
        set2 = set(col_list2)

        print('In columns1 but not in columns2:', sorted(set1 - set2))
        print('In columns2 but not in columns1:', sorted(set2 - set1))
        print('len(columns1):', len(col_list1))
        print('len(columns2):', len(col_list2))

        # stop
        return

    def debug_missing_columns(self, metrics):
        col_list1 = []
        col_list2 = []
        for col1 in self.historical_trades.columns:
            col_list1.append(col1)
        logging.info(
f'''
||| Before recording historical data:
||| {col_list1}
||| len(col_list1):
||| {len(col_list1)}
'''
        )
        self.record_historical_data(metrics)
        for col2 in self.historical_trades.columns:
            col_list2.append(col2)
        logging.info( 
f'''
||| After recording historical data:
||| {col_list2}
||| len(col_list2):
||| {len(col_list2)}
''' 
        )
        set1 = set(col_list1)
        set2 = set(col_list2)

        print('In columns1 but not in columns2:', sorted(set1 - set2))
        print('In columns2 but not in columns1:', sorted(set2 - set1))
        print('len(columns1):', len(col_list1))
        print('len(columns2):', len(col_list2))

        # stop
        return

    def get_init_state(self):
        float_values = self.scaler.transform(
            self.data.set_index('timestamp', drop=True, inplace=False).iloc[self.current_step-1].replace([-np.inf, np.inf], np.nan).fillna(value=0).values.reshape(1, -1))

        obs = self.running_scaler.transform(self.historical_trades.loc[
            self.previous_timestamp
            ].replace(
                [-np.inf, np.inf], np.nan
                ).fillna(
                    value=0
                    ).values.astype(np.float32).reshape(1, -1))

        obs = list(
            obs.flatten().astype(np.float32) # Original.
            ) + list(
                float_values.flatten().astype(np.float32)
                ) 
        obs_ = np.array(obs, dtype=np.float32)
        return obs_

    def get_last_valid_state(self):
        float_values = self.scaler.transform(
            self.data.set_index(
                'timestamp', drop=True, inplace=False
                ).iloc[self.current_step-1].replace(
                    [-np.inf, np.inf], np.nan
                    ).fillna(value=0).values.reshape(1, -1)
            )
        obs = self.running_scaler.transform(self.historical_trades.loc[
            self.previous_timestamp
            ].replace(
                [-np.inf, np.inf], np.nan
                ).fillna(
                    value=0
                    ).values.astype(np.float32).reshape(1, -1)
            )
        
        obs = list(
            obs.flatten().astype(np.float32) # Original.
            ) + list(
                float_values.flatten().astype(np.float32)
                )
        obs_ = np.array(obs, dtype=np.float32)
        return obs_

    def get_seq_state(self, window_size=7):
        
        # This will hold a list of (feature_dim,) arrays
        obs_seq = []

        # Get the indices for the sequence
        start_idx = max(self.start_step, self.current_step - window_size + 1)
        indices = range(start_idx, self.current_step + 1)

        # Loop through each time index, get the feature vector
        for t in indices:
            # ----- 1. Market features (scaled) -----
            float_values = self.data.set_index('timestamp', drop=True, inplace=False
                        ).iloc[t].replace([-np.inf, np.inf], np.nan
                        ).dropna().values.reshape(1, -1)
            float_values = self.scaler.transform(float_values)

            # ----- 2. Agent features (scaled) -----
            timestamp = self.data.iloc[t]['timestamp']
            if timestamp in self.historical_trades.index:
                hist_row = self.historical_trades.loc[timestamp].replace(
                    [-np.inf, np.inf], np.nan
                ).fillna(0).values.astype(np.float32).reshape(1, -1)
            else:
                # Use zeros if missing
                hist_row = np.zeros(
                    (1, self.historical_trades.shape[1]), dtype=np.float32
                    )
            hist_row = self.running_scaler.transform(hist_row)

            obs = np.concatenate([hist_row.flatten(), float_values.flatten()])
            obs_seq.append(obs)

            # hist_row = self.historical_trades.iloc[t].replace(
            #     [-np.inf, np.inf], np.nan
            # ).fillna(0).values.astype(np.float32).reshape(1, -1)
            # hist_row = self.running_scaler.transform(hist_row)

            # ----- 3. Concatenate -----
            obs = np.concatenate([hist_row.flatten(), float_values.flatten()])
            obs_seq.append(obs)

        # If sequence is shorter than window_size, pad with earliest available
        while len(obs_seq) < window_size:
            obs_seq.insert(0, obs_seq[0].copy())

        # Stack as array and add batch dimension
        obs_seq = np.stack(obs_seq)            # shape: (window_size, feature_dim)
        obs_seq = np.expand_dims(obs_seq, 0)   # shape: (1(batch_size), window_size(sequence_length), feature_dim)
        obs_seq = obs_seq.astype(np.float32)

        return obs_seq  # ready for LSTM: (batch=1, seq_len=window_size, features)

    
    def _get_obs(self):
        # Get the last 'window_size' rows from the raw data
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        # window_data = self.df_raw.iloc[start:end].values
        
        # Now transform with the scaler that was fit on historical data only
        # window_scaled = self.scaler.transform(window_data)
        # return window_scaled.flatten().astype(np.float32)

        # window_data = self.df_raw.set_index('timestamp', drop=True, inplace=False).iloc[start:end].fillna(value=0).replace([-np.inf, np.inf], np.nan).dropna().values
        window_data = self.df_raw.set_index('timestamp', drop=True, inplace=False).iloc[self.current_step].fillna(value=0).replace([-np.inf, np.inf], np.nan).dropna().values

        window_scaled = self.scaler.transform(window_data.reshape(1, -1))
        return window_scaled.flatten().astype(np.float32)

    @property
    def hrp_portfolio_value(self):
        return float(float(self.hrp_btc_quantity * self.hrp_btc_price) + float(self.hrp_eth_price * self.hrp_eth_quantity) + float(self.hrp_sp500_price * self.hrp_sp500_quantity))

    def get_scaler(self):
        scaler = deepcopy(self.scaler)
        return scaler

    def set_scaler(self, scaler):
        self.scaler = deepcopy(scaler)

    def get_running_scaler(self):
        running_scaler = deepcopy(self.running_scaler)
        return running_scaler

    def set_running_scaler(self, running_scaler):
        self.running_scaler = deepcopy(running_scaler)

 

    def calculate_vwap(self, asset):
        """Calculate the Volume Weighted Average Price (VWAP) for the given asset."""        
        
        print(f'self.historical_trades is: {self.historical_trades}')
        grouped = self.historical_trades.groupby('asset')
        print(f'grouped is {grouped}')# ||| self.historical_trades is: {self.historical_trades}')
        vwap = grouped.apply(lambda x: (x['price'] * x['volume']).cumsum() / x['volume'].cumsum())


        print(f'vwap is {vwap}')
        return vwap


    @staticmethod
    def softmax_normalization(actions):
        """
        Normalize the actions using the softmax function.

        Parameters:
        ----------
        actions : np.array
            The actions to be normalized.

        Returns:
        -------
        softmax_output : np.array
            The normalized actions.
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output


    def get_current_prices(self):
        """Get the current prices for selected assets."""
        current_prices = self.price_data.iloc[self.current_step][self.selected_assets].to_dict()
        return current_prices

    def get_previous_prices(self):
        """Get the previous prices for selected assets."""
        previous_prices = self.price_data.iloc[self.current_step-1][self.selected_assets].to_dict()
        return previous_prices

    def get_current_prices_list(self):
        """Get the current prices for selected assets."""
        current_prices = self.price_data.loc[self.timestamp][self.selected_assets].tolist()
        return current_prices

    def get_previous_prices_list(self):
        """Get the previous prices for selected assets."""
        previous_prices = self.price_data.loc[self.previous_timestamp][self.selected_assets].tolist()
        return previous_prices

    def get_next_prices_list(self):
        """Get the current prices for selected assets."""
        current_prices = self.price_data.loc[self.next_timestamp][self.selected_assets].tolist()
        return current_prices



    # Define function to calculate the auto-covariance matrix between pairs of columns
    @staticmethod
    def compute_auto_cov_matrix(df, lag=1):
        n = df.shape[1]
        auto_cov_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                series_i = df.iloc[:, i] - df.iloc[:, i].mean()
                series_j = df.iloc[:, j] - df.iloc[:, j].mean()
                auto_cov_matrix[i, j] = np.dot(
                    series_i[:-lag], series_j[lag:]
                    ) / len(series_i)
        
        return auto_cov_matrix

    def create_auto_cov_matrix(self, df, lag=1):
        # Select only numeric columns
        # numeric_data = data.select_dtypes(include=[np.number])
        
        # Compute the auto-covariance matrix for the provided data
        auto_cov_matrix = self.compute_auto_cov_matrix(
            # numeric_data,
            df,
            lag=lag
            )
        return auto_cov_matrix


    @staticmethod
    def auto_correlation_matrix(series, lag=1):
        """Compute the auto-correlation matrix for a single series with a given lag."""
        autocorr = np.corrcoef(
            np.array(
                [
                    series[:-lag], series[lag:]
                    ]
                )
        )
        return autocorr

    def compute_auto_correlation_matrix(
        self, df, lag=1
        ):
        """Compute the auto-correlation matrix for each pair of columns in the DataFrame."""
        n = df.shape[1]
        auto_corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Compute auto-correlation for the same column
                    auto_corr_matrix[i, j] = self.auto_correlation_matrix(
                        series=df.iloc[:, i].values, lag=lag
                        )[0, 1]
                else:
                    # For off-diagonal elements, consider pairwise correlations
                    series_i = df.iloc[:, i]
                    series_j = df.iloc[:, j]
                    auto_corr_matrix[i, j] = np.corrcoef(
                        series_i.shift(lag).fillna(value=0), # .dropna(), 
                        series_j.shift(lag).fillna(value=0) # .dropna()
                    )[0, 1]
        
        return auto_corr_matrix


    @staticmethod
    def rolling_window(arr, window):
        shape = arr.shape[:-1] + (
            arr.shape[-1] - window + 1, window
            )
        strides = arr.strides + (
            arr.strides[-1],
            )
        return np.lib.stride_tricks.as_strided(
            arr, shape=shape, strides=strides
            )

    def get_state(self, window_size=7, debug=False):


        float_values = self.data.set_index('timestamp', drop=True, inplace=False).iloc[self.current_step].replace([-np.inf, np.inf], np.nan).dropna().values.reshape(1, -1) # .tolist()

        if self.use_2d:
            return (state, l2_normalize(row=current_data))
        else:
            if self.current_step == self.start_step and self.initialized == False:

                obs_ = self.get_init_state()
            else:
                observation = self.historical_trades.loc[self.timestamp].replace([-np.inf, np.inf], np.nan).fillna(value=0).astype(np.float32).values # .reshape(1, -1)
                if debug:
                    row = self.historical_trades.loc[self.timestamp]
                    # print(f'observation: {observation}')
                    # print(f' np.max(observation) : {np.max(observation)}')
                    print("\n--- Value Check for row @", self.timestamp, "---")
                    for col, val in row.items():
                        if pd.isna(val):
                            print(f"[NaN] {col} = {val}")
                        elif np.isinf(val):
                            print(f"[Inf] {col} = {val}")
                        elif val == 0:
                            print(f"[Zero] {col} = {val}")
                        elif val > 1e4:
                            print(f"[Large] {col} = {val}")
                        elif val < -1e4:
                            print(f"[Small] {col} = {val}")

                try:
                    obs = self.running_scaler.transform(observation.reshape(1, -1))
                    if debug:
                        print(f' np.max(obs) : {np.max(obs)}')
                        print(f' np.min(obs) : {np.min(obs)}')
                except ValueError as e:
                    # find bad entries
                    bad_mask = ~np.isfinite(observation)
                    bad_idx  = np.where(bad_mask)[0]
                    print("Overflow or non-finite at positions:", bad_idx, "values:", observation[bad_mask])
                    # optionally force-clean again or skip this step

                    row = self.historical_trades.loc[self.timestamp]
                    obs = row.values  # your NumPy array

                    # find the “bad” positions again
                    bad_pos = np.where(~np.isfinite(obs))[0]
                    # grab the column names at those positions
                    bad_cols = row.index[bad_pos]
                    print("Columns with non-finite values at this timestamp:")
                    for col in bad_cols:
                        print(col)


                    col_list = list(self.historical_trades.columns)
                    print(f'col_list: {col_list} ||| len(col_list) : {len(col_list)}')
                    for idx in bad_idx:
                        print(f'{col_list[idx]}')
                    raise





                try:
                    observation = list(obs.flatten().astype(np.float32))
                except ValueError as e:
                    # find bad entries
                    # bad_mask = ~np.isfinite(observation)
                    bad_mask = np.max(observation)
                    bad_idx  = np.where(bad_mask)[0]
                    print("Overflow or non-finite at positions:", bad_idx, "values:", observation[bad_mask])
                    # optionally force-clean again or skip this step

                    # optionally force-clean again or skip this step

                    row = self.historical_trades.loc[self.timestamp]
                    obs = row.values  # your NumPy array

                    # find the “bad” positions again
                    bad_pos = np.where(np.max(obs))[0]
                    # grab the column names at those positions
                    bad_cols = row.index[bad_pos]
                    print("Columns with non-finite values at this timestamp:")
                    for col in bad_cols:
                        print(col)


                    col_list = list(self.historical_trades.columns)
                    print(f'col_list: {col_list} ||| len(col_list) : {len(col_list)}')
                    for idx in bad_idx:
                        print(f'{col_list[idx]}')
                    raise




                float_values = self.scaler.transform(float_values) # Original.

                observation += list(float_values.flatten().astype(np.float32))
                obs_ = np.array(observation, dtype=np.float32)


            return obs_



    def append_corr_matrix(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Appends the sliding window correlation matrix to the original multidimensional time series data.
        """
        # print('>>>>> Appending the correlation matrix <<<<<')

        # Generate column names for the flattened correlation matrix.
        columns = [
            '{}/{}'.format(m, n) for m, n in itertools.combinations(data.columns, 2)
            ]

        # Initialize an empty DataFrame to store flattened correlation matrices.
        corr_flattened = pd.DataFrame(index=data.index[window-1:], columns=columns)
        
        # Iterate through the DataFrame to compute the correlation matrices.
        for i in range(window - 1, len(data)):
            # Select the windowed subset of the data
            windowed_data = data.iloc[i-window+1:i+1]
            # print(f'windowed_data: {windowed_data} windowed_data.shape: {windowed_data.shape}')
            
            # Compute the correlation matrix for the current window.
            corr_matrix = windowed_data.corr().fillna(0)
            # print(f'corr_matrix: {corr_matrix} corr_matrix.shape: {corr_matrix.shape}')

            # Extract the upper triangular part of the correlation matrix, excluding the diagonal.
            flat_corr = corr_matrix.where(
                np.triu(
                    np.ones(corr_matrix.shape), k=1).astype(np.bool_)
                ).stack()
            # print(f'flat_corr: {flat_corr} ||| flat_corr.shape: {flat_corr.shape}')

            # Assign the flattened correlation values to the corresponding row in `corr_flattened`.
            corr_flattened.iloc[
                i - (window - 1)
                ] = flat_corr.values.flatten()
        
        # Concatenate the original time series data with the flattened correlation matrices.
        result = pd.concat(
            [
                data.iloc[window - 1:].reset_index(drop=True),
                corr_flattened.reset_index(drop=True)
                ], axis=1
            )
        return result

    @property
    def benchmark_return(self):
        if self.previous_hrp_portfolio_value > 1e-8:
            return float(float(
                self.hrp_portfolio_value - self.previous_hrp_portfolio_value
                ) / float(self.previous_hrp_portfolio_value))
        else:
            return float(0.)

    @property
    def market_return(self):
    # def risk_free_rate(self):
        return float(float(
            self.sp500_buy_hold - self.prev_sp500_buy_hold
            ) / float(self.prev_sp500_buy_hold))

    @property
    def risk_free_rate(self):
    # def market_return(self):
        return float(float(
            self.btc_buy_hold - self.prev_btc_buy_hold
            ) / self.prev_btc_buy_hold)

    # @lazyproperty
    @property
    def btc_buy_hold(self):
        init_volume = 1000 / self.price_data.iloc[self.start_step]['Close']
        return float(init_volume * self.price_data.iloc[self.current_step]['Close'])

    @property
    def prev_btc_buy_hold(self):
        init_volume = 1000 / self.price_data.iloc[self.start_step]['Close']
        return float(init_volume * self.price_data.iloc[self.current_step-1]['Close'])

    # @lazyproperty
    @property
    def eth_buy_hold(self):
        init_volume = 1000 / self.price_data.iloc[self.start_step]['ETH Close']
        # print(f"||| self.current_step: {self.current_step} self.price_data: {self.price_data} ||| init_volume: {init_volume} ||| self.price_data.iloc[self.current_step]['ETH Close']: {self.price_data.iloc[self.current_step]['ETH Close']}")
        # stop
        return float(init_volume * self.price_data.iloc[self.current_step]['ETH Close'])

    @property
    def prev_eth_buy_hold(self):
        init_volume = 1000 / self.price_data.iloc[self.start_step]['ETH Close']
        return float(init_volume * self.price_data.iloc[self.current_step-1]['ETH Close'])

    # @lazyproperty
    @property
    def sp500_buy_hold(self):
        init_volume = 1000 / self.price_data.iloc[self.start_step]['SP500 Close']
        return float(init_volume * self.price_data.iloc[self.current_step]['SP500 Close'])

    @property
    def prev_sp500_buy_hold(self):
        init_volume = 1000 / self.price_data.iloc[self.start_step]['SP500 Close']
        return float(init_volume * self.price_data.iloc[self.current_step-1]['SP500 Close'])

    # @lazyproperty
    @property
    def start_timestamp(self):
        return self.data.iloc[self.start_step]['timestamp']

    # @lazyproperty
    @property
    def end_timestamp(self):
        return self.data.iloc[self.start_step]['timestamp']
    
    # @lazyproperty
    @property
    def current_timestamp(self):
        assert self.data.iloc[self.current_step]['timestamp'] ==  self.price_data.iloc[self.current_step]['timestamp']
        return self.data.iloc[self.current_step]['timestamp']

    

    def is_month_end(self):
        current_date = pd.to_datetime(self.data.iloc[self.current_step]['timestamp'], unit='s')
        next_date = pd.to_datetime(self.data.iloc[self.current_step + 1]['timestamp'], unit='s') if self.current_step + 1 <= len(self.data) else None
        # self.rewards_sum += self.reward

        if next_date and current_date.month != next_date.month:
            self.is_today_month_end = True
            # Month has ended
            if self.portfolio_value >= self.btc_buy_hold: # and self.portfolio_value >= self.eth_buy_hold and self.portfolio_value >= self.sp500_buy_hold:
            # if float(self.portfolio_value - max(self.btc_buy_hold, self.eth_buy_hold, self.sp500_buy_hold)) >= 0.:
                self.is_successful_month_end = True
                return False # done = False
            else:
                # monthly_rewards_sum = np.sum(self.rewards) # -(self.rewards_sum)
                # if monthly_rewards_sum >= 0.:
                #     self.rewards_sum = -float(monthly_rewards_sum / 2)
                # else:
                #     self.rewards_sum = float(monthly_rewards_sum / 2)
                self.is_successful_month_end = False
                return True # done = True
        self.is_today_month_end = False
        self.is_successful_month_end = False
        return False

    # Helper function to check for leap years
    def is_leap_year(self, year):
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return True
        return False

    # @lazyproperty
    @property
    def timestamp(self):
        # print(f'||| self.current_step: {self.current_step} |||')
        # assert self.data.iloc[self.current_step]['timestamp'] ==  self.price_data.iloc[self.current_step]['timestamp'], f"||| {self.data.iloc[self.current_step]['timestamp']} |||  {self.price_data.iloc[self.current_step]['timestamp']}"
        return self.data.iloc[self.current_step]['timestamp']

    # @lazyproperty
    @property
    def previous_timestamp(self):
        # if self.start_step == self.current_step:
        #     return self.data.iloc[self.current_step]['timestamp']
        # else:
        #     # assert self.data.iloc[self.current_step-1]['timestamp'] ==  self.data.iloc[self.current_step-1]['timestamp'], f"||| {self.data.iloc[self.current_step-1]['timestamp']} |||  {self.price_data.iloc[self.current_step-1]['timestamp']}"
        return self.data.iloc[self.current_step-1]['timestamp']

    @property
    def prev_previous_timestamp(self):
        return self.data.iloc[self.current_step-2]['timestamp']

    @property
    def next_timestamp(self):
        return self.data.iloc[self.current_step+1]['timestamp']


    def get_current_price(self):
        # Assuming self.data is a DataFrame with prices and self.current_step is the current timestep
        return self.price_data.at[self.current_step, self.price_column]

    def get_days_until_month_end(self, date):
        next_month = date.replace(day=28) + timedelta(days=4)  # this will never fail
        month_end = next_month - timedelta(days=next_month.day)
        return float((month_end - date).days)
    
    def get_days_since_month_start(self, date):
        month_start = date.replace(day=1)
        return float((date - month_start).days)



    def get_days_until_month_end(self, date):
        # Convert the timestamp to a datetime object if not already
        timestamp = deepcopy(date)
        next_month = date.replace(day=28) + timedelta(days=4)  # this will never fail
        month_end = next_month - timedelta(days=next_month.day)
        return float((month_end - date).days)

    def get_days_since_month_start(self, date):
        # Convert the timestamp to a datetime object if not already
        timestamp = deepcopy(date)
        month_start = timestamp.replace(day=1)
        return float((timestamp - month_start).days)

    @property
    def days_until_month_end(self): # days_since_month_start
        # next_month = deepcopy(self.timestamp).replace(day=28) + timedelta(days=4)  # this will never fail
        # month_end =  deepcopy(self.timestamp).replace(day=28) + timedelta(days=4) - timedelta(days=next_month.day)
        return float(
            (
                (
                    deepcopy(self.timestamp).replace(day=28) + timedelta(days=4) - timedelta(
                        days=(
                            deepcopy(self.timestamp).replace(day=28) + timedelta(days=4)).day
                        )
                    ) - deepcopy(self.timestamp)
                ).days
            )

    # @lazyproperty
    @property
    def days_since_month_start(self):
        # month_start = deepcopy(self.timestamp).replace(day=1)
        return float((deepcopy(self.timestamp) - deepcopy(self.timestamp).replace(day=1)).days)

    # @lazyproperty
    @property
    def hrp_get_current_weights(self):
        weights = []
        weights += [self.hrp_balance / self.hrp_portfolio_value]
        weights += [self.hrp_btc_value / self.hrp_portfolio_value]
        weights += [self.hrp_eth_value / self.hrp_portfolio_value]
        weights += [self.hrp_sp500_value / self.hrp_portfolio_value]
        # print(f'weights: {weights}')
        return weights


    @property
    def get_current_weights(self):


        portfolio_value = self.portfolio_value
        weights = []
        weights += [self.balance / portfolio_value]
        weights += [self.btc_value / portfolio_value]
        weights += [self.eth_value / portfolio_value]
        weights += [self.sp500_value / portfolio_value]
        # print(f'weights: {weights}')
        return weights

    def initalize_portfolio_weights(self):
        # self.balance = self.initial_balance
        self.current_step = self.start_step
        self.check_returns = 0  # Resetting to an empty list
        self.reward = 0
        self.btc_quantity = 0
        self.btc_price = 0
        self.eth_price = 0
        self.sp500_price = 0
        self.action = 0
        self.returns.clear() #  = []
        self.rewards = []
        self.eth_quantity = 0
        self.sp500_quantity = 0
        self.check_returns = 0
        self.btc_quantity = 0
        self.price = 0
        self.action = 0
        
        self.actions = [0, 0, 0] # self.action_space.sample() # [0, 0, 0] # Hard coded as of now. In range of len(self,selected_assets).
        self.hrp_actions = [0, 0, 0] 
        self.btc_value = self.btc_price * self.btc_quantity
        self.eth_value = self.eth_price * self.eth_quantity
        self.sp500_value = self.sp500_price * self.sp500_quantity

        actions_ = self.action_space.sample()
        if not self.use_2d: # and self.current_step == self.start_step:
            raw_actions = actions_
            action_weights = actions_
            # print(f'action_weights: {action_weights}')
            action_weights = self.softmax_normalization(actions=action_weights).tolist()
            weights = self.get_current_weights[1:]  # Ensure this is a tensor or list as required
            actions = [float(aw) - float(w) for aw, w in zip(action_weights, weights)]
            self.execute_trade(actions)
        else:
            actions = actions.squeeze().tolist()

    def check_done(self, actions: List[float]) -> Tuple[bool, bool, str]:
        current_prices = self.get_current_prices()
        
        '''if self.current_step >= self.end_step - 1:
            return True, True, "end_step"  # Termination due to reaching the end step
        if self.is_month_end():
            if self.current_step < self.end_step - 1:
                return True, False, "successful_month"  # Successful month end but not the end of the trading period
            return True, True, "end_of_period"  # End of the trading period
        
        if self.balance < 0.:
            return True, False, "negative_balance"  # Truncation due to negative balance
        if self.portfolio_value < 0:
            return True, False, "negative_portfolio_value"  # Truncation due to negative portfolio value
        
        for asset, action in zip(self.selected_assets, actions):
            if -1 < action < 0:
                if not (self.portfolio[asset] * current_prices[asset] > 10):
                    return True, False, "insufficient_asset_value_for_selling"  # Truncation due to insufficient asset value for selling
            elif 0 < action < 1:
                if not (self.portfolio[asset] * current_prices[asset] > 10):
                    return True, False, "insufficient_asset_value_for_buying"  # Truncation due to insufficient asset value for buying
            elif action < -1 or action > 1:
                return True, False, "action_out_of_bounds"  # Truncation due to action out of bounds'''
        # '# Truncation due to negative asset value' '# Truncation due to negative asset value' '# Truncation due to insufficient cash' 
        # '# Truncation due to insufficient balance' 'Truncation due to action out of bounds' 'Truncation due to insufficient asset value for buying' 
        # 'Truncation due to insufficient asset value for selling' 'Truncation due to negative portfolio value' 'Truncation due to negative balance' 
        # End of the trading period''# Successful month end but not the end of the trading period.' '# Termination due to reaching the end step' "negative_asset_value"  
        # Truncation due to negative asset value  # Truncation due to insufficient cash # Truncation due to insufficient balance
        if self.balance < 0: # <= 10.:
            # return True, False, "insufficient_balance"  # Truncation due to insufficient balance
            print("Truncation due to insufficient balance")
            return True or False
        # if self.portfolio.get('cash', 0) < 0: # <= 10.:
        #     # return True, False, "insufficient_cash"  # Truncation due to insufficient cash
        #     print("Truncation due to insufficient cash")
        #     return True or False
        if any(self.portfolio[asset] < 0. for asset in self.selected_assets):
            # return True, False, "negative_asset_value"  # Truncation due to negative asset value
            print("Truncation due to negative asset value")
            return True or False
        if self.current_step >= self.end_step - 1:
            # return True, True,  '# Termination due to reaching the end step'
            print('# Termination due to reaching the end step')
            return True or True
        # if self.is_month_end():
        #     if self.current_step < self.end_step - 2:
        #         print('Fail to meet month end obligations') # Successful month end but not the end of the trading period. ||| self.is_month_end() but not self.current_step >= self.end_step - 2')
        #         return False or True
        #         # return False, True, '# Successful month end but not the end of the trading period.'
        #     print('# End of the trading period!!!')
        #     return True or True
            # return True, True, '# End of the trading period'
        if self.balance < 0.:
            print('Truncation due to negative balance')
            return True or False
            # return True, False, 'Truncation due to negative balance'
        if self.portfolio_value < 0: 
            print('Truncation due to negative portfolio value')
            return True or False
            # return True, False, 'Truncation due to negative portfolio value'
        self.positive_actions = 0
        # assert self.positive_actions == 0, f'self.positive_actions: {self.positive_actions}' #  ||| action: {action}'
        # print(f'||| actions: {actions}')
        for asset, action in zip(self.selected_assets, actions):
            # print(f'||| action: {action}') # self.current_step += 1
            if -1 <= action < 0:
                # if not self.portfolio[asset] * current_prices[asset] > action * self.portfolio[asset] >= 0 and self.portfolio[asset] * current_prices[asset] - (action * self.portfolio[asset]) > 10:
                if not float(abs(action) * self.portfolio[asset] * current_prices[asset]) >= 10.: # or self.portfolio[asset] * current_prices[asset] - abs(action * self.portfolio[asset] * current_prices[asset]) >= 0.):
                # if not (self.portfolio[asset] * current_prices[asset] - (action * self.portfolio[asset]) > 10):
                    # print(f"Truncation due to insufficient asset value for selling ||| abs(action * self.portfolio[asset] * current_prices[asset])/Intended Sale of assets has to be >= 10.: {abs(action * self.portfolio[asset] * current_prices[asset])} ||| self.portfolio[asset] * current_prices[asset]/Current Value of the asset: {self.portfolio[asset] * current_prices[asset]} ||| abs(action * self.portfolio[asset] * current_prices[asset])/Intended allocations: {abs(action * self.portfolio[asset] * current_prices[asset])} ||| Would've Achieved Allocations/ self.portfolio[asset] * current_prices[asset] - abs(action * self.portfolio[asset] - * current_prices[asset]) has to >= 0: {self.portfolio[asset] * current_prices[asset] - abs(action * self.portfolio[asset] * current_prices[asset])}") # if not (self.portfolio[asset] * current_prices[asset] > 10):
                    # self.render()
                    # print(f'||| Truncation due to insufficient asset value for selling ||| asset: {asset} ||| action: {action} ||| self.portfolio[asset]: {self.portfolio[asset]} ||| current_prices[asset]: {current_prices[asset]} ||| self.portfolio[asset] * current_prices[asset]: {self.portfolio[asset] * current_prices[asset]}')
                    return True or False
                    # return True, False, 'Truncation due to insufficient asset value for selling'
            elif 0 < action <= 1:
                self.positive_actions += action
                # print(f'self.positive_actions: {self.positive_actions} ||| action: {action}')
                # if not (self.balance - (action * self.balance) >= 10 or action * self.balance >= 10.):
                if not abs(action) * self.balance >= 10.:
                # if not self.balance > action * current_prices[asset] * current_prices[asset] >= 10. or not self.balance - (action * current_prices[asset] * current_prices[asset]) >= 10:
                # if not self.balance - (action * current_prices[asset]) > 10):
                    # print(f'Truncation due to insufficient cash balance asset value for buying ||| self.balance: {self.balance} ||| abs(action * self.balance) / Intended aciton have to be >= 10.: action: {action} / {abs(action * self.balance)} ||| current_prices[asset]: {current_prices[asset]} ||| self.balance - (self.balance * action): {self.balance - (action * self.balance)} ||| current_prices[asset]: {current_prices[asset]}')
                    return True or False
            elif action < -1 or action > 1:
                # print('Truncation due to action out of bounds')
                return True or False
        # if not (self.positive_actions<=1):
            # self.positive_actions = 0
            # return True, False, f'Suspect Truncation due to insufficient asset value for buying! positive_actions: {self.positive_actions}'
        # self.positive_actions = 0
        # if self.balance <= 10.:
        #     print('# Truncation due to insufficient balance | balance <= 0.')
        #     return True or False
        # if self.portfolio.get('cash', 0) <= 10.:
        #     print('# Truncation due to insufficient cash')
        #     return True or False
        if any(self.portfolio[asset] < 0. for asset in self.selected_assets):
            print('# Truncation due to negative asset value')
            return True or False

        # return False, False  # No termination or truncation
        # print("No termination or truncation")
        # print("Good!!!!!!!!!!")
        return False or False

    def compute_returns(self, profit: float, previous_profit: float):
        """
        Compute both the simple and log returns between two profit values.
        - delta_profit: (profit - previous_profit) / previous_profit, or 0 if previous_profit == 0
        - log_delta_profit: signed log-return, nan if previous_profit == 0
        """
        if previous_profit != 0.0:
            # delta_profit = (profit - previous_profit) / previous_profit # Original.
            delta_profit = (profit - previous_profit) / abs(previous_profit)
            log_delta_profit = self.signed_log_return(profit=profit, prev_profit=previous_profit)
        else:
            delta_profit = 0.0
            log_delta_profit = 0.0 # np.nan

        return float(delta_profit), float(log_delta_profit)

    # Example helper from above
    @staticmethod
    def signed_log_return(profit: float, prev_profit: float) -> float:
        if prev_profit == 0:
            return np.nan
        ratio = profit / prev_profit
        # log(|ratio|) is always defined when prev_profit != 0
        return np.sign(ratio) * np.log(abs(ratio))

    def step(self, actions, debug=False):
        self.prev_should_be_portfolio_value = self.should_be_portfolio_values[-1] if len(self.should_be_portfolio_values) > 0 else float(self.should_be_portfolio_value)
        self.previous_hrp_portfolio_value = self.hrp_portfolio_value
        self.previous_actions = self.actions
        actions = actions.squeeze()
        self.previous_portfolio_value = self.portfolio_values[-1]
        self.raw_actions = actions
        self.previous_profit = self.profit
        self.prev_prev_balance = self.prev_balance
        self.prev_balance = self.balance
        self.hrp_prev_balance = self.hrp_balance

        self.prev_r_ = self.r_
        self.prev_prospect_theory_loss_aversion_return = self.prospect_theory_loss_aversion_return

        self.prev_max_drawdown = self.max_drawdown
        self.prev_var = self.var
        self.prev_parametric_var = self.parametric_var
        self.prev_parametric_var_dollar = self.prev_parametric_var
        self.prev_cvar = self.cvar
        self.prev_parametric_cvar = self.parametric_cvar # self.param_cvar
        self.prev_cvar_dollar = self.cvar_dollar
        self.prev_return_ = self.return_
        self.previous_weights = self.weights_history[-1].tolist() if len(self.weights_history) > 0 else self.current_weights

        for asset in self.selected_assets:
            self.previous_portfolio[asset] = self.portfolio[asset]
        self.previous_cash = float(self.portfolio['cash'])
        total_previous_portfolio_value = self.get_no_actions_portfolio_value()
        returns = self.returns_df.iloc[:self.current_step][self.selected_assets]

        # Initialize the HRP optimizer
        hrp_optimizer = HierarchicalRiskParity(returns) # , asset_names=self.selected_assets)

        # Compute HRP weights using Expected Shortfall (CVaR)
        hrp_weights, order, hrp_weights_series = hrp_optimizer.get_hrp_weights() # risk_measure='cvar', confidence_level=0.05) # asset_names=list(self.returns_df.columns), asset_prices=self.price_data.set_index('timestamp', drop=True).iloc[:self.current_step][self.selected_assets], risk_measure='expected_shortfall',
     
        # Step 1: Create a mapping from HRP indices to the original order
        index_mapping = {
            hrp_idx: orig_idx for orig_idx, hrp_idx in enumerate(order)
            }
        # print(f'index_mapping: {index_mapping}')

        # Step 2: Reorder the HRP weights to match selected_assets' order
        hrp_weights_ordered = [
            hrp_weights[index_mapping[i]] for i in range(len(order))
            ]

        # Convert back to numpy array (optional)
        hrp_weights_ordered = np.array(hrp_weights_ordered).flatten().astype(np.float32).tolist()
        

        self.raw_hrp_actions = hrp_weights_ordered

        hrp_current_weights = self.hrp_get_current_weights # [1:]  # Ensure this is a tensor or list as required
        hrp_weights = hrp_current_weights[1:]
        self.hrp_previous_weights = hrp_current_weights
        hrp_actions = [float(aw) - float(w) for aw, w in zip(hrp_weights_ordered, hrp_weights)]
        self.hrp_actions = hrp_actions

        # Constructing the dictionary
        hrp_actions_dict = {
            asset: hrp_actions[i] for i, asset in enumerate(
                self.selected_assets
                )
            }


        previous_value = float(
            self.balance + sum(
                self.portfolio[asset] * self.get_previous_prices()[asset] for asset in self.selected_assets
                )
            )

        if isinstance(actions, List):
            actions = np.array(actions, dtype=np.float32).squeeze()

        if self.current_step == self.start_step and self.use_2d:
            actions = actions.squeeze().tolist()
        else:
            if not self.use_2d: # and self.current_step == self.start_step:
                self.raw_actions = actions
                if self.use_action_norm:
                    action_weights = actions.squeeze().tolist()
                    self.raw_actions = deepcopy(action_weights)
                    action_weights = self.softmax_normalization(actions=action_weights).tolist()
                    weights = self.get_current_weights[1:]  # Ensure this is a tensor or list as required
                    actions = [float(aw) - float(w) for aw, w in zip(action_weights, weights)]
                    self.actions = actions
                    self.action_weights = actions
            else:
                actions = actions.squeeze().tolist()


      
        if not self.use_action_norm:
            self.actions = actions # Original.
        # is_done = self.evalueate_execute_trade(actions=self.actions)
        # done = self.current_step >= self.end_step - 1
        # is_month_end_ = self.is_month_end()
        if self.train:
            done = (self.current_step >= self.end_step - 1 or
                    # self.check_done(actions=actions.squeeze().tolist()) or
                    # self.check_done(actions=actions) or
                    # self.evalueate_execute_trade(actions=actions) or
                    # is_done or
                    self.balance < 0. or self.portfolio_value <= 0 or
                    # self.is_month_end() or
                    any(self.portfolio[asset] < 0. for asset in self.selected_assets))
        else:
            done = (self.current_step >= self.end_step - 1 or
                    # self.check_done(actions=actions.squeeze().tolist()) or
                    # self.check_done(actions=actions) or
                    # self.evalueate_execute_trade(actions=actions) or
                    # is_done or
                    self.balance < 0. or self.portfolio_value <= 0 or
                    # self.is_month_end() or
                    any(self.portfolio[asset] < 0. for asset in self.selected_assets))


        self.done = done
        if not done:
 
            self.total_transaction_fee = 0
            blocked_counts = self.execute_trade(actions, hrp_actions=hrp_actions) # Original.


            # 2) Update the Dirichlet posterior
            self.block_tracker.track(blocked_counts)


            self.current_weights = self.get_current_weights
            if debug:
                print(f'||| current_weights: {self.current_weights} |||') #  current_weights.shape: {current_weights.shape}')
            self.update(portfolio_value=float(self.portfolio_value), weights=np.array(self.current_weights), benchmark_return=self.benchmark_return) # Original.

            current_weights = self.get_current_weights
            hrp_current_weights = self.hrp_get_current_weights[1:]
            hrp_previous_weights = self.hrp_previous_weights[1:]
            current_prices = self.get_current_prices_dict()
            previous_prices = self.get_previous_prices_dict()
            previous_weights = self.previous_weights
            current_prices_list = self.get_current_prices_list()
            names_list = []
            asset_weights = current_weights[1:]
            previous_asset_weights = self.previous_weights[1:]

            hrp_prev_asset_quantity_list = []
            hrp_current_asset_quantity_list = []
            for asset in self.selected_assets:
                hrp_prev_asset_quantity_list.append(
                    float(
                        self.historical_trades.at[
                            self.previous_timestamp,
                            f'hrp_portfolio_{asset}_owned'
                            ]
                        )
                    )

                hrp_current_asset_quantity_list.append(
                    float(
                        self.hrp_portfolio.get(f'{asset}', 0.0)

                        )
                    )

            hrp_should_be_asset_value = [
                p_q * p_p for p_q, p_p in zip(
                    hrp_prev_asset_quantity_list, current_prices_list
                    )
                ]

            hrp_current_asset_value = [
                p_q * p_p for p_q, p_p in zip(
                    hrp_current_asset_quantity_list, current_prices_list
                    )
                ]

            hrp_asset_returns = [
                float(float(
                hrp_current_asset_value[i] - hrp_should_be_asset_value[i]
                ) / hrp_should_be_asset_value[i]) if hrp_should_be_asset_value[i] != 0. else 0. for i in range(len(self.selected_assets))
                ]
            for i in range(len(self.selected_assets)):
                asset_value = float(
                    self.portfolio_value * asset_weights[i]
                    )
                previous_asset_value = float(
                    self.previous_portfolio_value * previous_asset_weights[i]
                    )
                hrp_asset_value = float(
                    self.hrp_portfolio_value * hrp_current_weights[i]
                    )
                hrp_previous_asset_value = float(
                    self.previous_hrp_portfolio_value * hrp_previous_weights[i]
                    )
                # 2) During each step of your env loop, update each engine
                #    (you’ll plug in your real values for asset_value, weights, etc.)
                for name, fe in self.fe_dict.items():
                    asset = self.selected_assets[i]
                    fe.update(
                        asset_value=asset_value,
                        weight=asset_weights[i],
                        current_price=current_prices[asset] if asset != 'cash' else float(1.), # self.historical_trades.at[self.timestamp, 'cash_balance']
                        previous_price=previous_prices[asset] if asset != 'cash' else float(1.),
                        # previous_price=previous_prices[i] if asset != 'cash' else float(self.historical_trades.at[self.timestamp, 'previous_cash_balance']),
                        previous_weight=previous_asset_weights[i],
                        current_weight=asset_weights[i],
                        prev_asset_quantity=float(
                            self.historical_trades.at[
                                self.previous_timestamp, f'portfolio_{asset}_owned'
                                ]
                            ) if asset != 'cash' else float(self.prev_balance),
                        # prev_balance=self.prev_balance,
                        prev_balance=self.prev_prev_balance,
                        # weight=current_weights[i],
                        cash=self.portfolio['cash'],
                        # benchmark_return=float(float(hrp_asset_value - float(hrp_previous_asset_value)) / float(hrp_previous_asset_value)) if hrp_previous_asset_value != 0. else 0., # float(float(hrp_asset_value - float(self.portfolio_value * current_weights[i]) / self.initial_balance)),
                        # benchmark_return=float(self.benchmark_return),
                        # benchmark_return=float(float(hrp_current_asset_value[i] - hrp_should_be_asset_value[i]) / hrp_should_be_asset_value[i]) if hrp_should_be_asset_value[i] != 0. else 0.,
                        benchmark_return=float(hrp_asset_returns[i]), # -1] if i!=0 else self.hrp_portfolio['cash']),
                        debug=debug
                    )
                   
            self.hrp_portfolio_values.append(self.hrp_portfolio_value)

            if len(self.portfolio_values) >= 3:
                 prev_portfolio_value = self.portfolio_values[-2] # if len(self.cumulative_returns_history) >= 2 else self.cumulative_returns_history[-1]
                 profit_margin_ = self.portfolio_value - prev_portfolio_value
                 # profit_margin = (profit_margin_ / prev_portfolio_value) - 1  # Normalize reward
                 profit_margin = profit_margin_ / prev_portfolio_value
            else:
                #self.portfolio = portfolio_value
                profit_margin_ = self.portfolio_value - self.initial_balance
                # profit_margin = (profit_margin_ / self.initial_balance) - 1  # Normalize reward
                profit_margin = profit_margin_ / self.initial_balance
                prev_portfolio_value = self.initial_balance
            '''
            # Typical values:
            α = 1.5
            α=1.5 to 3.0

            3.0 is common for introducing moderate to strong asymmetry.
            α=2:
            α=2 matches classic loss aversion settings from Prospect Theory.
            How high can you set alpha?
            There's no hard upper bound, but too high (
            α>5): may cause learning instability, as losses will dominate.
            α=1:
            α=1 is baseline, # similar to a log return
            α>1:
            α>1 introduces penalty; tune via validation or cross-validation.
            '''
            prospect_theory_loss_aversion_return, log_return = self.calculate_immediate_reward(profit_margin, alpha=self.reward_weights["prospect_theory_loss_aversion_alpha"])
            self.prospect_theory_loss_aversion_return = prospect_theory_loss_aversion_return
            self.cumulative_returns_history.append(log_return)
            #print(f'cumulative_return is: {cumulative_return}')
            # Update reward calculation
            current_prices = self.get_current_prices()
            profit = 0
            for asset in self.selected_assets:
                # fifo_total = self.live_pnl.get_total_pnl(
                #     f'{asset}', current_price=current_prices.get(f'{asset}', 0), method="FIFO"
                #     )
                # lifo_total = self.live_pnl.get_total_pnl(
                #     f'{asset}', current_price=current_prices.get(f'{asset}', 0), method="LIFO"
                #     )
                # profits += self.live_pnl.get_total_pnl(f'{asset}', current_price=current_prices.get(f'{asset}'), method=self.method)
                # self.rewards.append(reward_)
                # print(f"Live assets PnL (FIFO): {fifo_total:.2f} | Live BTC PnL (LIFO): {lifo_total:.2f}\n")

                trade_price = current_prices[asset]
                # evaluation = self.evaluate_asset_trade(asset, trade_price, self.timestamp)
                # print(f'||| {asset}: evaluation: {evaluation} |||')
                # if evaluation["delta_pnl"] is not None:
                #     profit += evaluation["delta_pnl"]

                evaluation = self.evaluate_asset_trade(asset, trade_price, self.timestamp)
                # print(f'||| {asset}: evaluation: {evaluation} |||')
                if evaluation["delta_pnl"] is None:
                    delta_pnl = self.historical_trades.at[self.timestamp, f"{asset}_PnL"]
                    self.historical_trades.at[self.timestamp, f"{asset}_PnL"] = delta_pnl # float(0.)
                    # print(f'||| {asset} : evaluation: {evaluation} |||')
                else:
                # if evaluation["delta_pnl"] is not None:
                    profit += evaluation["delta_pnl"]
                    # Save the trade evaluation in the DataFrame for later reference.
                    self.historical_trades.at[self.timestamp, f"{asset}_PnL"] = evaluation["delta_pnl"]
                    # print(f'||| {asset} : evaluation: {evaluation} |||')
            if self.train == False and debug:
                print(f'||| self.actions: {self.actions} ||| profit: {profit} ||| self.portfolio_value : {self.portfolio_value} ||| self.previous_portfolio_value : {self.previous_portfolio_value} ||| ||| self.balance: {self.balance}')

            
            self.profits.append(profit)
            self.profit = profit
            portfolio_value = self.portfolio_values[-1]

            pct_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
            self.historical_pct_returns.append(pct_return)


            max_drawdown = self.calculate_max_drawdown()
            var = self.calculate_var(confidence_level=0.95)
            parametric_var = self.calculate_parametric_var(confidence_level=0.95)

            # Historical CVaR (default)
            hist_cvar = calculate_cvar(
                data=self.returns, confidence_level=0.95, method="historical", input_is_returns=True
                )

            # Parametric CVaR
            param_cvar = calculate_cvar(
                data=self.returns, confidence_level=0.95, method="parametric", input_is_returns=True
                )


            cvar_dollar = -(portfolio_value * abs(param_cvar))

            parametric_var_dollar = portfolio_value * parametric_var

            self.max_drawdown = max_drawdown
            self.var = var
            self.parametric_var = parametric_var
            self.cvar = hist_cvar
            self.parametric_cvar = param_cvar
            self.cvar_dollar = cvar_dollar
            self.parametric_var_dollar = parametric_var_dollar

            portfolio_values = np.array(self.portfolio_values)
            returns_ = portfolio_values[1:] / portfolio_values[:-1] - 1

            # Or log-returns:
            # returns_ = np.diff(np.log(portfolio_values))
            return_ = returns_[-1]
            self.return_ = return_
 
            if self.current_step - self.start_step >= 3: # self.seq_len:
                metrics = self.compute_metrics() # Original.
            else:
                metrics = self.metrics

            # 3) At any time (e.g. end of episode or for logging), pull out all metrics
            if self.current_step - self.start_step >= 3:
                latest_metrics = {}
                for name, fe in self.fe_dict.items():
                    latest = fe.compute_metrics()
                    latest_metrics[name] = latest  # or latest
                for name, val in latest_metrics.items():
                    for key, val in latest_metrics[name].items():
                        metrics.update({key: latest_metrics[name][key]})

            self.record_historical_data(metrics) # Original.


            # beta_adj_sharpe = float(self.historical_trades.at[self.timestamp, "pf_beta_adj_sharpe"]) # Secondary: weight ~= 0.1<->0.2
            m_squared_ratio = float(float(self.historical_trades.at[self.timestamp, "pf_m_squared_ratio"])) # Secondary: weight ~= 0.1<->0.2
            alpha_monthly = float(self.historical_trades.at[self.timestamp, "pf_alpha_monthly"]) # Primary: weight ~= 1.0
            vol_monthly = float(self.historical_trades.at[self.timestamp, "pf_vol_monthly"]) # Penalty: ~= 0.1<->0.5
            # pf_cvar_05 = float(self.historical_trades.at[self.timestamp, "pf_cvar_05"]) # Penalty: ~= 0.1<->0.5
            pf_sharpe = float(self.historical_trades.at[self.timestamp, "pf_sharpe"]) # Primary: weight ~= 1.0
            pf_sortino = float(self.historical_trades.at[self.timestamp, "pf_sortino"]) # Secondary: weight ~= 0.1<->0.2
            pf_max_drawdown = float(self.historical_trades.at[self.timestamp, "pf_max_drawdown"]) # Penalty: ~= 0.1<->0.5
            pf_beta = float(self.historical_trades.at[self.timestamp, "pf_beta"]) # Secondary: weight ~= 0.1<->0.2
            # skew_ = float(self.historical_trades.at[self.timestamp, "pf_skew"]) # Skew: # Bouns: ~= 0.01<->0.1
            # kurtosis_ = float(self.historical_trades.at[self.timestamp, "pf_kurtosis"]) # Kurtosis # Penalty: ~= 0.01<->0.05
            # pf_sortino = float(self.historical_trades.at[self.timestamp, "sortino"]) # Secondary: weight ~= 0.1<->0.2


            def calculate_pct_delta_return(prev_value, current_value):
                if prev_value != 0.:
                    if prev_value < 0. or current_value < 0.:
                        return float(float(current_value - prev_value) / abs(prev_value))
                    else: # both positive numbers.
                        return float(float(current_value - prev_value) / prev_value)
                else:
                    return 0.

            
            current_prices = self.get_current_prices_list()
            previous_prices = self.get_previous_prices_list()
            # print(f'||| current_prices: {current_prices} ||| previous_prices: {previous_prices} |||')
            previous_weights = self.previous_weights[1:]
            current_weights = self.get_current_weights[1:]
            delta = [float(float(p_t - prev_p) / prev_p) for p_t, prev_p in zip(current_prices, previous_prices)]

            asset_quantity_list = []
            prev_asset_quantity_list = []
            for asset in self.selected_assets:
                # prev_asset_quantity_list.append(float(self.historical_trades.at[self.prev_previous_timestamp, f'portfolio_{asset}_owned']))
                prev_asset_quantity_list.append(float(self.historical_trades.at[self.previous_timestamp, f'portfolio_{asset}_owned']))
                # asset_quantity_list.append(float(self.historical_trades.at[self.timestamp, f'portfolio_{asset}_owned']))
                asset_quantity_list.append(self.portfolio[f'{asset}'])

            # should_be_portfolio_value = self.prev_balance + sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, previous_prices)])
            # should_be_portfolio_value = self.prev_prev_balance + np.sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, current_prices)])
            should_be_portfolio_value = self.prev_balance + np.sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, current_prices)])
            # next_portfolio_value = self.balance + np.sum([p_q * p_p for p_q, p_p in zip(asset_quantity_list, next_prices)])
            # portfolio_value = self.balance + sum([p_q * p_p for p_q, p_p in zip(asset_quantity_list, current_prices)])
            # should_be_portfolio_value = np.sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, current_prices)])
            # portfolio_value = np.sum([p_q * p_p for p_q, p_p in zip(asset_quantity_list, current_prices)])
            self.should_be_portfolio_value = should_be_portfolio_value
            self.should_be_portfolio_values.append(should_be_portfolio_value)
            sharpe = float(self.historical_trades.at[self.timestamp, "sharpe"])
            # print(f'||| sharpe: {sharpe} |||')
            # base_reward = (next_portfolio_value - should_be_portfolio_value) / should_be_portfolio_value if should_be_portfolio_value != 0. else 0.
            base_reward = (self.portfolio_value - should_be_portfolio_value) / self.previous_portfolio_value if self.previous_portfolio_value != 0. else 0.
            # base_reward = float(float(portfolio_value - should_be_portfolio_value) / should_be_portfolio_value) if should_be_portfolio_value !=0. else 0.
            reward = base_reward
            # reward = float(self.portfolio_value - should_be_portfolio_value) / self.previous_portfolio_value if self.previous_portfolio_value != 0. else 0. # Original and Works.
            # log_return_ = np.log(next_portfolio_value / should_be_portfolio_value) if self.previous_portfolio_value > 0. and should_be_portfolio_value > 0. else 0.
            log_return_ = np.log(self.portfolio_value / should_be_portfolio_value) if self.previous_portfolio_value > 0. and should_be_portfolio_value > 0. else 0.
            self.log_return = log_return_
            reward += log_return_
            reward += sharpe
            # reward = self.reward_weights['base_ret'] * reward
            if debug:
                print(
f'''
||| base_reward: {base_reward}
||| self.portfolio_value: {self.portfolio_value}
||| should_be_portfolio_value: {should_be_portfolio_value}
||| log_return_: {log_return_}
||| self.balance: {self.balance}
||| self.prev_balance: {self.prev_balance}
||| self.prev_prev_balance: {self.prev_prev_balance}
||| sharpe: {sharpe}
||| reward: {reward}
||| current_prices: {current_prices}
||| prev_asset_quantity_list: {prev_asset_quantity_list}
||| hrp_asset_returns: {hrp_asset_returns}
'''
                )
            self.reward = reward
            self.rewards.append(reward)
            # print(f"||| pf_beta_adj_sharpe: {self.historical_trades.at[self.timestamp, 'pf_beta_adj_sharpe']} ||| reward: {reward} |||")
            # ||| metrics: {'sharpe': 0.0, 'sortino': 0.0, 'momentum': -0.002095800000000092, 'skewness': 0.0, 'kurtosis': 0.0, 'realized_vol': 0.0, 'ulcer_index': 0.0, 'vol_of_vol': 0.0, 'beta': 0.0, 'correlation': 0.0, 'omega': 0.0, 'calmar_mean_r': >


            if debug:
                row = self.historical_trades.loc[self.timestamp]
                print("\n--- Value Check for row @", self.timestamp, "---")
                for col, val in row.items():
                    if pd.isna(val):
                        print(f"[NaN] {col} = {val}")
                    elif np.isinf(val):
                        print(f"[Inf] {col} = {val}")
                    elif val == 0:
                        print(f"[Zero] {col} = {val}")



            if not self.use_seq_obs:
                obs = self.get_state()
            else:
                obs = self.get_seq_state(window_size=self.seq_len)

            self.current_step += 1

            # Ensure standard Gym return types
            obs = np.asarray(obs, dtype=np.float32)
            return obs, float(reward), bool(done), {}
        else: 
         
            if self.current_step >= self.end_step - 1:
                if self.use_action_norm:
                    # print(f'actions: {actions}')
                    if not isinstance(actions, List):
                        action_weights = actions.squeeze().tolist()
                    else:
                        action_weights = actions
                    self.raw_actions = deepcopy(action_weights)
                    # print(f'action_weights: {action_weights}')

                    action_weights = self.softmax_normalization(actions=action_weights).tolist()
                    # print(f'action_weights: {action_weights}')
                    weights = self.get_current_weights[1:]  # Ensure this is a tensor or list as required
                    # print(f"||| weights: {weights} ")
                    actions = [float(aw) - float(w) for aw, w in zip(action_weights, weights)]
                    # actions = [float(aw) - float(w) for aw, w in zip(actions, weights)]
                    self.actions = actions
                    self.action_weights = actions

                self.total_transaction_fee = 0
                blocked_counts = self.execute_trade(actions, hrp_actions=hrp_actions) # Original.
                
                # 2) Update the Dirichlet posterior
                self.block_tracker.track(blocked_counts)

                # self.actions = realigned_actions # action
                if not self.use_action_norm:
                    self.actions = actions # Original.

                self.current_weights = self.get_current_weights
                # print(f'||| self.current_weights: {self.current_weights} ||| self.previous_weights: {self.previous_weights}')
                if debug:
                    print(f'||| current_weights: {self.current_weights}') #  ||| current_weights.shape: {current_weights.shape}') 
                self.update(portfolio_value=float(self.portfolio_value), weights=np.array(self.current_weights), benchmark_return=self.benchmark_return)

                # self.portfolio_values.append(self.portfolio_value)
                self.hrp_portfolio_values.append(self.hrp_portfolio_value)

                if self.train and self.current_step >= self.end_step - 1:
                    self.historical_trades.to_csv('sac_historical_trades_train.csv')
                elif not self.train and self.current_step >= self.end_step - 1:
                    self.historical_trades.to_csv('sac_historical_trades_test.csv')


                if len(self.portfolio_values) >= 2:
                     prev_portfolio_value = self.portfolio_values[-2] # if len(self.cumulative_returns_history) >= 2 else self.cumulative_returns_history[-1]
                     profit_margin_ = self.portfolio_value - prev_portfolio_value
                     profit_margin = profit_margin_ / prev_portfolio_value
                else:
                    profit_margin_ = self.portfolio_value - self.initial_balance
                    profit_margin = profit_margin_ / self.initial_balance
                    prev_portfolio_value = self.initial_balance

                '''
                # Typical values:
                α = 1.5
                α=1.5 to 3.0

                3.0 is common for introducing moderate to strong asymmetry.
                α=2:
                α=2 matches classic loss aversion settings from Prospect Theory.
                How high can you set alpha?
                There's no hard upper bound, but too high (
                α>5): may cause learning instability, as losses will dominate.
                α=1:
                α=1 is baseline, # similar to a log return
                α>1:
                α>1 introduces penalty; tune via validation or cross-validation.
                '''
                # prospect_theory_loss_aversion_return, log_return = self.calculate_immediate_reward(profit_margin, alpha=2.0)
                prospect_theory_loss_aversion_return, log_return = self.calculate_immediate_reward(profit_margin, alpha=self.reward_weights["prospect_theory_loss_aversion_alpha"])
                self.cumulative_returns_history.append(log_return)
                self.prospect_theory_loss_aversion_return = prospect_theory_loss_aversion_return

                # reward = self.calculate_reward(immediate_cumulative_return, prev_portfolio_value) #tanh_of_cumulative_return
                current_prices = self.get_current_prices()
                # reward = 0
                profit = 0
                for asset in self.selected_assets:
                    # fifo_total = self.live_pnl.get_total_pnl(
                    #     f'{asset}', current_price=current_prices.get(f'{asset}'), method="FIFO"
                    #     )
                    # lifo_total = self.live_pnl.get_total_pnl(
                    #     f'{asset}', current_price=current_prices.get(f'{asset}'), method="LIFO"
                    #     )
                    # reward += self.live_pnl.get_total_pnl(f'{asset}', current_price=current_prices.get(f'{asset}'), method=self.method)
                    # profits += self.live_pnl.get_total_pnl(f'{asset}', current_price=current_prices.get(f'{asset}'), method=self.method)

                    trade_price = current_prices[asset]
                    # evaluation = self.evaluate_asset_trade(asset, trade_price, self.timestamp)
                    # print(f'||| {asset}: evaluation: {evaluation} |||')
                    # profit += evaluation["delta_pnl"]

                    # self.drag_forward_historical_trade(asset, timestamp=self.timestamp)
                    evaluation = self.evaluate_asset_trade(asset, trade_price, self.timestamp) 
                    if evaluation["delta_pnl"] is None:
                        delta_pnl = self.historical_trades.at[self.timestamp, f"{asset}_PnL"]
                        self.historical_trades.at[self.timestamp, f"{asset}_PnL"] = delta_pnl # float(0.)
                        # print(f'||| {asset} : evaluation: {evaluation} |||')
                    else:
                    # if evaluation["delta_pnl"] is not None:
                        profit += evaluation["delta_pnl"]
                        # Save the trade evaluation in the DataFrame for later reference.
                        self.historical_trades.at[self.timestamp, f"{asset}_PnL"] = evaluation["delta_pnl"]
                        # print(f'||| {asset} : evaluation: {evaluation} |||')




                self.profits.append(profit)
                self.profit = profit
                portfolio_value = self.portfolio_values[-1]

                pct_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
                self.historical_pct_returns.append(pct_return)

                max_drawdown = self.calculate_max_drawdown()
                var = self.calculate_var(confidence_level=0.95)
                parametric_var = self.calculate_parametric_var(confidence_level=0.95)

                var = self.portfolio_value * var
                parametric_var = self.portfolio_value * parametric_var



                # Historical CVaR (default)
                hist_cvar = calculate_cvar(
                    data=self.returns, confidence_level=0.95, method="historical", input_is_returns=True
                    )

                # Parametric CVaR
                param_cvar = calculate_cvar(
                    data=self.returns, confidence_level=0.95, method="parametric", input_is_returns=True
                    )


                # Dollar-Value Conversion
                # If you want dollar-value expected shortfall:
                portfolio_value = self.portfolio_values[-1]
                cvar_dollar = -(portfolio_value * abs(param_cvar))

                self.max_drawdown = max_drawdown
                self.var = var
                self.parametric_var = parametric_var
                self.cvar = hist_cvar
                self.parametric_cvar = param_cvar
                self.cvar_dollar = cvar_dollar

                portfolio_values = np.array(self.portfolio_values)
                returns_ = portfolio_values[1:] / portfolio_values[:-1] - 1

                # Or log-returns:
                # returns_ = np.diff(np.log(portfolio_values))
                return_ = returns_[-1]
                self.return_ = return_

                # print(f'||| max_drawdown: {max_drawdown} ||| reward: {reward} ||| var: {var} ||| parametric_var: {parametric_var} ||| self.prev_cvar_dollar: {self.prev_cvar_dollar} ||| cvar_dollar: {cvar_dollar} ||| profits: {profits} ||| self.previous_profits: {self.previous_profits} ||| gain: {gain} ||| total_previous_portfolio_value: {total_previous_portfolio_value} ||| return_: {return_} ||| param_cvar: {param_cvar}')
                

                if self.current_step - self.start_step >= 3:
                    metrics = self.compute_metrics()
                else:
                    metrics = self.metrics
                if debug:
                    print(f'||| metrics: {metrics} |||')

                self.record_historical_data(metrics) # original.


                # beta_adj_sharpe = float(self.historical_trades.at[self.timestamp, "pf_beta_adj_sharpe"]) # Secondary: weight ~= 0.1<->0.2
                # m_squared_ratio = float(self.historical_trades.at[self.timestamp, "pf_m_squared_ratio"]) # Secondary: weight ~= 0.1<->0.2
                m_squared_ratio = float(float(self.historical_trades.at[self.timestamp, "pf_m_squared_ratio"])) # Secondary: weight ~= 0.1<->0.2
                alpha_monthly = float(self.historical_trades.at[self.timestamp, "pf_alpha_monthly"]) # Primary: weight ~= 1.0
                vol_monthly = float(self.historical_trades.at[self.timestamp, "pf_vol_monthly"]) # Penalty: ~= 0.1<->0.5
                # Skew: # Bouns: ~= 0.01<->0.1
                # Kurtosis # Penalty: ~= 0.01<->0.05
                # Total_Transaction_Costs: # Penalty: ~= 0.001<->0.01
                # pf_cvar_05 = float(self.historical_trades.at[self.timestamp, "pf_cvar_05"]) # Penalty: ~= 0.1<->0.5
                pf_sharpe = float(self.historical_trades.at[self.timestamp, "pf_sharpe"]) # Primary: weight ~= 1.0
                # pf_sortino = float(self.historical_trades.at[self.timestamp, "pf_sortino"]) # Secondary: weight ~= 0.1<->0.2
                pf_max_drawdown = float(self.historical_trades.at[self.timestamp, "pf_max_drawdown"]) # Penalty: ~= 0.1<->0.5
                pf_beta = float(self.historical_trades.at[self.timestamp, "pf_beta"]) # Secondary: weight ~= 0.1<->0.2
                # skew_ = float(self.historical_trades.at[self.timestamp, "pf_skew"])
                # kurtosis_ = float(self.historical_trades.at[self.timestamp, "pf_kurtosis"])
                # pf_sortino = float(self.historical_trades.at[self.timestamp, "sortino"])


                
                current_prices = self.get_current_prices_list()
                previous_prices = self.get_previous_prices_list()
                # next_prices = self.get_next_prices_list()
                previous_weights = self.previous_weights[1:]
                current_weights = self.get_current_weights[1:]
                delta = [float(float(p_t - prev_p) / prev_p) for p_t, prev_p in zip(current_prices, previous_prices)]

                prev_asset_quantity_list = []
                asset_quantity_list = []
                for asset in self.selected_assets:
                    prev_asset_quantity_list.append(float(self.historical_trades.at[self.previous_timestamp, f'portfolio_{asset}_owned']))
                    asset_quantity_list.append(self.portfolio[f'{asset}'])

                # next_portfolio_value = self.balance + np.sum([p_q * p_p for p_q, p_p in zip(asset_quantity_list, next_prices)])
                # should_be_portfolio_value = self.prev_prev_balance + np.sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, current_prices)])
                should_be_portfolio_value = self.prev_balance + np.sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, current_prices)])
                # should_be_portfolio_value = self.prev_balance + sum([p_q * p_p for p_q, p_p in zip(prev_asset_quantity_list, previous_prices)])

                sharpe = float(self.historical_trades.at[self.timestamp, "sharpe"])
                self.should_be_portfolio_value = should_be_portfolio_value
                self.should_be_portfolio_values.append(should_be_portfolio_value)
                # print(f'||| sharpe: {sharpe} |||')
                # reward = (self.portfolio_value - should_be_portfolio_value) / should_be_portfolio_value if should_be_portfolio_value != 0. else 0.
                reward = (self.portfolio_value - should_be_portfolio_value) / self.previous_portfolio_value if self.previous_portfolio_value != 0. else 0.
                # reward = float(self.portfolio_value - should_be_portfolio_value) / self.previous_portfolio_value if self.previous_portfolio_value != 0. else 0. # Original and Works.
                log_return_ = np.log(self.portfolio_value / should_be_portfolio_value) if self.previous_portfolio_value > 0. and should_be_portfolio_value > 0. else 0.

                self.log_return = log_return_
                reward += log_return_
                reward += sharpe
                # print(f'||| should_be_portfolio_value: {should_be_portfolio_value} ||| self.portfolio_value: {self.portfolio_value} |||')


                def calculate_pct_delta_return(prev_value, current_value):
                    if prev_value != 0.:
                        return float(float(current_value - prev_value) / abs(prev_value))
                    else:
                        return 0.


            
                '''reward = float(float(self.reward_weights['pf_sharpe']) * delta_pf_sharpe) - float(float(self.reward_weights['vol_monthly']) * delta_vol_monthly) 
                reward += float(float(self.reward_weights['log_return']) * log_return) - float(float(self.reward_weights['pf_max_drawdown']) * delta_pf_max_drawdown) 
                reward += float(float(self.reward_weights['pf_cvar_05']) * delta_pf_cvar_05)
                reward += float(float(self.reward_weights['alpha_monthly']) * delta_alpha_monthly)
                # reward += delta_profit
                reward += float(float(self.reward_weights['m_squared_ratio']) * delta_m_squared_ratio)
                # reward += float(float(self.reward_weights['beta_adj_sharpe']) * delta_beta_adj_sharpe)
                reward += float(float(self.reward_weights['pf_sortino']) * delta_pf_sortino)
                reward -= float(float(self.reward_weights['pf_beta']) * delta_pf_beta)
                reward += float(self.reward_weights["prospect_theory_loss_aversion_w"] * delta_prospect_theory_loss_aversion_return)'''


                '''p_mean       = self.block_tracker.posterior_mean()
                p_var        = self.block_tracker.posterior_variance()
                p_ent        = self.block_tracker.posterior_entropy()
                # r_ = -float(0.1 * p_var.sum())
                # r_ -= float(0.01 * p_ent)
                # r_ += float(0.05 * p_mean.sum())
                r_ = -float(p_var.sum())
                r_ -= float(p_ent)
                r_ += float(p_mean.sum())
                # reward -= float(float(self.reward_weights['blocked_actions_w']) * r_) # Original.
                # Total_Transaction_Costs: # Penalty: ~= 0.001<->0.01
                reward += float(float(self.reward_weights['tx_fee_w']) * self.total_transaction_fee)

                delta_r_ = calculate_pct_delta_return(self.prev_r_, self.r_)
                reward -= float(float(self.reward_weights['blocked_actions_w']) * delta_r_)
                # reward -= delta_r_
                self.r_ = r_
                self.prospect_theory_loss_aversion_return = prospect_theory_loss_aversion_return

                self.reward = reward
                self.rewards.append(reward)'''
                
                if not self.use_seq_obs:
                    obs = self.get_state()
                else:
                    obs = self.get_seq_state(window_size=self.seq_len)

                self.current_step += 1
                obs = np.asarray(obs, dtype=np.float32)
                return obs, float(reward), bool(done), {}

            if self.use_2d:
                return (self.state, self.obs), 0.0, bool(done), {}
            else:
                if done and not self.is_month_end():
                    if self.train:
                        self.historical_trades.to_csv('ppo_historical_trades_train.csv')
                    else:
                        self.historical_trades.to_csv('ppo_historical_trades_test.csv')

               
                if not self.use_seq_obs:
                    obs = self.get_last_valid_state()
                else:
                    obs = self.get_seq_state(window_size=self.seq_len)

                obs = np.asarray(obs, dtype=np.float32)
                return obs, 0.0, bool(done), {}

    def calculate_var(
        self,
        # returns,
        confidence_level=0.95
        ):
        """
        Calculate Value at Risk (VaR) using historical simulation.

        Args:
            returns (list or np.array): Historical returns, as decimals (e.g., 0.01 for 1%)
            confidence_level (float): Confidence level for VaR (default: 0.95)

        Returns:
            float: The VaR value (as a negative number, e.g., -0.02 for -2%)
        """
        portfolio_values = np.array(self.portfolio_values)
        returns = portfolio_values[1:] / portfolio_values[:-1] - 1

        # Or log-returns:
        # returns = np.diff(np.log(portfolio_values))

        returns = np.asarray(returns)
        # if len(returns) < 2:
        if len(portfolio_values) < 2:
            return 0.0  # Not enough data

        # Historical VaR: e.g., 5th percentile for 95% VaR
        percentile = 100 * (1 - confidence_level)
        var = np.percentile(returns, percentile)
        return var

   
    def calculate_immediate_reward(self, profit_margin, alpha=1.0): # Tune alpha > 1 to make negatives harsher. #, action, profit_margin):
        '''
        # Typical values:
        α = 1.5
        α=1.5 to 3.0

        3.0 is common for introducing moderate to strong asymmetry.
        α=2:
        α=2 matches classic loss aversion settings from Prospect Theory.
        How high can you set alpha?
        There's no hard upper bound, but too high (
        α>5): may cause learning instability, as losses will dominate.
        α=1:
        α=1 is baseline, # similar to a log return
        α>1:
        α>1 introduces penalty; tune via validation or cross-validation.

        | Theory          | Field     | Used for                                   | Typical Loss Aversion (λ or α) |
        | --------------- | --------- | ------------------------------------------ | ------------------------------ |
        | Prospect Theory | Economics | Modeling real decision-making under risk   | 2.0–2.5                        |
        | Utility Theory  | Finance   | Classical portfolio optimization           | 1.0 (symmetric)                |
        | RL w/ Asymmetry | Quant/RL  | Loss-averse rewards, robust trading agents | 1.5–3.0 (user-tuned)           |
        '''
        # Or:
        # Perhaps using the concepts from HER(Hind sight replay buffer's concept and the rewards accordingly)?

        #logging.info(f'  profit_margin is: {profit_margin}')
        #logging.info(f'  profit_margin is: {np.log(1 + abs(profit_margin)) * -2}')
        # Reward positive profits more conservatively than losses

        if self.portfolio_value == 0 or self.previous_portfolio_value == 0:
            return float(0.), float(0.)
        else:
            if self.current_step != self.start_step:
                assert self.portfolio_value > 0 and self.previous_portfolio_value > 0, f'Error in reward calculation! ||| {self.current_step} ||| {self.start_step} ||| self.previous_portfolio_value: {self.previous_portfolio_value} ||| self.portfolio_value: {self.portfolio_value} ||| self.balance : {self.balance} ||| self.portfolio: {self.portfolio} |||'
            log_ret = np.log(self.portfolio_value / self.previous_portfolio_value)
            # Use a “softplus” to smooth it:
            softplus_reward = np.log(1 + np.exp(log_ret)) - alpha * np.log(1 + np.exp(-log_ret))
            # Tune alpha > 1 to make negatives harsher
            # print(f'||| log_ret: {log_ret} ||| softplus_reward: {softplus_reward} |||')
            return softplus_reward, log_ret


    def calculate_pnl(self):
        """Calculate Profit and Loss based on historical trades and current market prices."""
        pnl = 0
        current_prices = self.current_prices
        for index, trade in self.historical_trades.iterrows():
            if trade[f'{asset}_trade_type'] == 'sell':
                # PnL for sell trades
                pnl += (trade[f'{asset}_price'] - self.calculate_vwap(trade['asset'])) * trade[f'{asset}_volume']
            elif trade[f'{asset}_trade_type'] == 'buy':
                # Subtract the cost of buy trades from PnL
                pnl -= (trade[f'{asset}_price'] - current_prices[trade['asset']]) * trade[f'{asset}_volume']
        return pnl
    

    @property
    def market_return(self):
        # return float(float(float(float(self.sp500_buy_hold - self.prev_sp500_buy_hold) / self.prev_sp500_buy_hold) + float(float(self.btc_buy_hold - self.prev_btc_buy_hold) / self.prev_btc_buy_hold) + float(float(self.eth_buy_hold - self.prev_eth_buy_hold) / self.prev_eth_buy_hold)) / 3)
        # return float(float(self.btc_buy_hold - self.prev_btc_buy_hold) / self.prev_btc_buy_hold)
        return float(float(self.sp500_buy_hold - self.prev_sp500_buy_hold) / self.prev_sp500_buy_hold)


    @property
    def risk_free_rate(self):
        # return float(float(self.btc_buy_hold - self.prev_btc_buy_hold) / self.prev_btc_buy_hold)
        return float(float(self.sp500_buy_hold - self.prev_sp500_buy_hold) / self.prev_sp500_buy_hold)

    @property
    def balance(self):
        cash_balance = float(self.portfolio.get('cash', 0.0))
        return cash_balance

    @property
    def hrp_balance(self):
        return float(self.hrp_portfolio.get('cash', 0.0))

    @property
    def btc_quantity(self):
        return float(self.portfolio.get('Close', 0.0))

    @property
    def eth_quantity(self):
        return float(self.portfolio.get('ETH Close', 0.0))

    @property
    def sp500_quantity(self):
        return float(self.portfolio.get('SP500 Close', 0.0))

    @property
    def btc_price(self):
        return float(self.price_data.iloc[self.current_step]['Close']) # float(self.price_data.at[self.current_step, 'ETH Close']) # float(self.price_data.iloc[self.current_step]['ETH Close'])

    @property
    def eth_price(self): 
        return float(self.price_data.iloc[self.current_step]['ETH Close']) # float(self.price_data.at[self.current_step, 'ETH Close'])

    @property
    def sp500_price(self): # self.price_data.at[self.timestamp, 'Close']
        return float(self.price_data.iloc[self.current_step]['SP500 Close']) # float(self.price_data.at[self.current_step, 'SP500 Close'])

    # @lazyproperty
    @property
    def btc_value(self):
        # assert float(
        #     self.btc_price * self.btc_quantity
        #     ) == float(
        #         self.portfolio['Close'] * self.btc_price
        #         ), f"{float(self.btc_price * self.btc_quantity)} vs. {float(self.portfolio['Close'] * self.btc_price)}"
        return self.btc_quantity * self.btc_price
        # return float(self.btc_price * self.btc_quantity)
    
    # @lazyproperty
    @property
    def eth_value(self):
        # assert float(
        #     self.eth_price * self.eth_quantity
        #     ) == float(
        #         self.portfolio['ETH Close'] * self.eth_price
        #         ), f"{float(self.eth_price * self.eth_quantity)} vs. {float(self.portfolio['ETH Close'] * self.eth_price)}"
        return self.eth_quantity * self.eth_price
        # return float(self.eth_price * self.eth_quantity)
    
    # @lazyproperty
    @property
    def sp500_value(self):
        # assert float(
        #     self.sp500_price * self.sp500_quantity
        #     ) == float(
        #         self.portfolio['SP500 Close'] * self.sp500_price
        #         ), f"{float(self.sp500_price * self.sp500_quantity)} vs. {float(self.portfolio['SP500 Close'] * self.sp500_price)}"
        return self.sp500_quantity * self.sp500_price

    # @lazyproperty
    @property
    def hrp_btc_value(self):
        assert float(
            self.hrp_btc_price * self.hrp_btc_quantity
            ) == float(
                self.hrp_portfolio['Close'] * self.hrp_btc_price
                ), f"{float(self.hrp_btc_price * self.hrp_btc_quantity)} vs. {float(self.hrp_portfolio['Close'] * self.hrp_btc_price)}"
        return self.hrp_portfolio['Close'] * self.hrp_btc_price
        # return float(self.hrp_btc_price * self.hrp_btc_quantity)
    
    @property
    def hrp_eth_value(self):
        assert float(
            self.hrp_eth_price * self.hrp_eth_quantity
            ) == float(
                self.hrp_portfolio['ETH Close'] * self.hrp_eth_price
                ), f"{float(self.hrp_eth_price * self.hrp_eth_quantity)} vs. {float(self.hrp_portfolio['ETH Close'] * self.hrp_eth_price)}"
        return self.hrp_portfolio['ETH Close'] * self.hrp_eth_price
        # return float(self.hrp_eth_price * self.hrp_eth_quantity)
    
    @property
    def hrp_sp500_value(self):
        assert float(
            self.hrp_sp500_price * self.hrp_sp500_quantity
            ) == float(
                self.hrp_portfolio['SP500 Close'] * self.hrp_sp500_price
                ), f"{float(self.hrp_sp500_price * self.hrp_sp500_quantity)} vs. {float(self.hrp_portfolio['SP500 Close'] * self.hrp_sp500_price)}"
        return self.hrp_portfolio['SP500 Close'] * self.hrp_sp500_price
    

    @property
    def portfolio_value(self):
        '''value = 0
        current_prices_list = self.get_current_prices_list()
        for key, i in zip(self.portfolio.keys(), current_prices_list):
            if key == 'cash':
                value += self.portfolio[key]
            else:
                print(f'i: {i} ||| key: {key}')
                value += self.portfolio[key] * current_prices_list[i]
        print(f' portfolio_value: {value}')
        return value'''
        # cash = self.balance # self.portfolio[asset]
        return float(self.balance + sum(self.portfolio[asset] * self.get_current_prices()[asset] for asset in self.selected_assets)) # + float(self.cash_reserve)
        # return float(cash + sum(self.portfolio[asset] * self.get_current_prices()[asset] for asset in self.selected_assets))

    @property
    def hrp_portfolio_value(self):
        return self.hrp_balance + sum(
            self.hrp_portfolio[asset] * self.get_current_prices()[asset] for asset in self.selected_assets
            )

    @staticmethod
    def sigmoid_sign(array, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5
        return sigmoid(array / thresh) * thresh

    # def markowitz_optimization_with_constraints(self, cov_matrix, expected_returns):
    def markowitz_optimization(self, cov_matrix, expected_returns):
        import cvxpy as cp
        # Convert to numpy for cvxpy processing
        cov_matrix_np = cov_matrix.squeeze().cpu().numpy()
        expected_returns_np = expected_returns.squeeze() # .cpu().numpy()

        n = expected_returns_np.shape[0]
        weights = cp.Variable(n)

        # Portfolio variance, which is a quadratic form
        portfolio_variance = cp.quad_form(weights, cov_matrix_np)
        expected_return = weights.T @ expected_returns_np

        # Objective: Maximize expected return adjusted by risk (variance)
        objective = cp.Maximize(expected_return - portfolio_variance)
        constraints = [cp.sum(weights) == 1,  # sum of weights is 1
                       weights >= 0]          # weights are non-negative

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Check if the problem was successfully solved
        if problem.status not in ["infeasible", "unbounded"]:
            # Successfully solved
            optimized_weights = weights.value
        else:
            # Problem was not solved, return equal weights or some fallback
            optimized_weights = np.ones(n) / n

        return torch.tensor(optimized_weights, dtype=torch.float32)


    def evalueate_execute_trade(self, actions):
        """
        Execute trades based on the actions array. A positive action means a buy,
        a negative action means a sell. After processing each asset’s trade, update that asset’s PnL.
        """
        try:
            current_prices = self.get_current_prices()
        except Exception as e:
            logger.error(
                "Cannot execute trade because current prices could not be retrieved: %s", str(e)
                )
            return
        # Iterate over the actions and corresponding assets
        # for i, proportion in enumerate(actions): # Original.
        for i in range(len(self.selected_assets)):
            asset = self.selected_assets[i]
            trade_price = current_prices[asset]
            proportion = float(actions[i])
            # print(
            #     f'||| asset: {asset} ||| trade_price: {trade_price} ||| proportion: {proportion} ||| i: {i} ||| actions: {actions}'
            #     )
            # proportion = float(actions[i])
            is_done = False
            # Buying logic
            if proportion > 0 and proportion < 1.:
            # if proportion > 0 and proportion <= 1 and self.balance > 10. and self.portfolio['cash'] > 10. and abs(proportion) * self.balance >= 10.:
                # abs(action * self.balance) / Intended aciton have to be >= 10.:
                effective_price = trade_price * (1 + self.transaction_fee)
                buy_amount = self.portfolio['cash'] * proportion / effective_price
                # cash_cost = buy_amount * effective_price # Original.
                cash_cost = self.portfolio['cash'] * proportion
                # print(f'||| Not enough balance to buy? ||| self.balance >= cash_cost >= 10.: {self.balance} >= {cash_cost} >= 10. ||| buy_amount: {buy_amount} > 10.?')
                if self.balance >= cash_cost >= 10. and buy_amount > 0.:
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_trade_amount'] = float(buy_amount)
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_cash_cost_revenue'] = -float(cash_cost)
                    # self.record_trade(
                    #     asset, volume, effective_price, "buy"
                    #     )
                    # self.live_pnl.process_trade(
                    #     asset, buy_amount, effective_price, "buy"
                    #     )
                    # logger.info(
                    #     "BUY executed for %s: volume=%.4f at effective price=%.4f; cost=%.4f", asset, buy_amount, effective_price, cash_cost
                    #     )
                    # logging.info(
                    #     "returning False"
                    #     )
                    is_done = False

                # elif self.balance >= cash_cost >= 0. and self.balance >= cash_cost <= 10. and buy_amount > 0. and self.balance >= cash_cost >= 10.:
                #     return False

                else:
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_trade_amount'] = float(buy_amount)
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_cash_cost_revenue'] = -float(cash_cost)
                    # print(
                    #     f'||| Not enough balance to buy? ||| self.balance >= cash_cost >= 0.: {self.balance} >= {cash_cost} >= 10. ||| buy_amount: {buy_amount} > 0. ||| proportion: {proportion}?'
                    #     )
                    # logger.warning(
                    #     "Not enough balance to buy %s: required=%.4f, available=%.4f", asset, cash_cost, self.balance
                    #     )
                    # logging.warning(
                    #     "returning True"
                    #     )
                    is_done = True
            elif proportion < 0 and proportion > -1.:
            # if proportion < 0 and proportion >= -1 and float(abs(proportion) * self.portfolio[asset] * trade_price) >= 10.:
                # abs(action * self.portfolio[asset] * current_prices[asset])/Intended Sale of assets has to be >= 10.:
                effective_price = trade_price * (1 - self.transaction_fee)
                sell_amount = self.portfolio[asset] * abs(proportion)
                cash_revenue = sell_amount * effective_price

                if self.portfolio[asset] >= sell_amount >= 0. and cash_revenue >= 10.:
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_trade_amount'] = -float(sell_amount)
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_cash_cost_revenue'] = float(cash_revenue)
                    # cash_cost_revenue_string = f'{asset}_cash_cost_revenue'
                    # print(f'||| asset: {asset}')
                    # print(f'||| sell_amount: {sell_amount} ||| cash_revenue: {cash_revenue} ||| self.historical_trades[{asset}_trade_amount]: {self.historical_trades[asset_trade_amount_string]} ||| self.historical_trades[{asset}_cash_cost_revenue]: {self.historical_trades[cash_cost_revenue_string]}')
                    # self.live_pnl.process_trade(
                    #     asset=asset, volume=-float(sell_amount), price=effective_price, trade_type="sell"
                    #     )
                    # logger.info(
                    #     "SELL executed for %s: volume=%.4f at effective price=%.4f; revenue=%.4f", asset, sell_amount, effective_price, cash_revenue
                    #     )
                    # logging.info(
                    #     "returning False"
                    #     )
                    is_done = False

                # elif cash_revenue >= 0. and cash_revenue <= 10. and self.portfolio[asset] >= sell_amount >= 0.:
                #     return True

                else:
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_trade_amount'] = -float(sell_amount)
                    self.historical_trades.at[
                        self.timestamp, f'{asset}_cash_cost_revenue'] = float(cash_revenue)
                    # print(f'||| Not enough balance to sell? ||| cash_revenue {cash_revenue} >= 10? ||| proportion: {proportion}? ||| sell_amount: {sell_amount} ||| self.portfolio[asset]: {self.portfolio[asset]}')
                    # logger.warning(
                    #      "Not enough holdings to sell %s: trying to sell %.4f, holdings=%.4f", asset, sell_amount, self.portfolio[asset]
                    #      )
                    # logging.warning(
                    #     "returning True"
                    #     )
                    is_done = True

            elif proportion == 0: # and proportion > -1.:
                is_done = False
            # else:
            #     logger.debug(
            #         f"No trade executed for %s (action=actions[i]). {asset}"
            #         )
            #     logging.warning(
            #         "returning True"
            #         )
            else:
                # logger.error(
                #     "Error executing trade for asset %s: %s", asset, str(e)
                #     )
                # else:
                # logger.debug(
                #     f"No trade executed for %s (action=actions[i]). {asset}"
                #     )
                # logging.warning(
                #     "returning True"
                #     )
                is_done = True

        return is_done

    


    def compute_transaction_fee(self, trade_value):
        """Returns total fee for a trade of given notional size."""
        return self.fixed_fee + self.proportional_fee * abs(trade_value)




    
    # def execute_trade_negative_first(self, actions, hrp_actions): # This method sorts the negative actions and perfroms them first!
    def execute_trade(self, actions, hrp_actions): # Original. Works!
        """
        Executes RL policy actions as portfolio trades with **SELLS before BUYS**.
        Handles:
          - Buy/sell, no trade if abs(notional) < min_trade_value after fees.
          - Fixed + proportional fees.
          - Recording all trades and summary stats.
          - PnL update (if live_pnl attached).
        """
        self.traded = 0
        try:
            current_prices = self.get_current_prices()
        except Exception as e:
            logger.error("Cannot execute trade: %s", str(e))
            return

        # Ensure row exists in historical_trades
        if self.timestamp not in self.historical_trades.index:
            self.historical_trades.loc[self.timestamp] = np.nan

        # Prepare actions with asset context
        trade_orders = []
        for i, proportion in enumerate(actions):
            asset = self.selected_assets[i]
            proportion = round(proportion, 4)
            trade_orders.append((asset, proportion)) # , hrp_actions[i]))
            hrp_proportion = hrp_actions[i]
            trade_price = current_prices[asset]
            # Always execute HRP trade logic as before
            self.execute_trade_dict(
                asset=asset, proportion=hrp_proportion, trade_price=trade_price
                )

        # **Split actions: Sell first, then buy, then zero**
        sells = [order for order in trade_orders if order[1] < 0]
        buys  = [order for order in trade_orders if order[1] > 0]
        # zeros can be handled or skipped
        # zeros = [order for order in trade_orders if order[1] == 0]
        """print(
f'''
||| trade_orders:
||| {trade_orders}
||| sells:
||| {sells}
||| buys:
||| {buys}
'''
        )"""

        # New accumulators per step
        portfolio_bought_value, portfolio_sold_value = 0.0, 0.0
        total_transaction_fee, portfolio_total_transactions_value = 0.0, 0.0

        # 1) Execute your trades, detect which assets got blocked:
        # blocked_counts = np.zeros(self.n_assets, int)

        # Helper for trade logic (remains unchanged)
        def process_trade(asset, proportion, trade_price, hrp_proportion): # , blocked_counts):
            nonlocal portfolio_bought_value, portfolio_sold_value
            nonlocal total_transaction_fee, portfolio_total_transactions_value
            nonlocal blocked_counts
            if trade_price == 0:
                return
            """print(
f'''
||| actions: {actions}
||| asset: {asset}
||| proportion: {proportion}
||| trade_orders:
||| {trade_orders}
||| sells: {sells}
||| buys: {buys}
'''
            )"""
            cash = self.portfolio['cash']
            position = self.portfolio[asset]

            # Always execute HRP trade logic as before
            # self.execute_trade_dict(
            #     asset=asset, proportion=hrp_proportion, trade_price=trade_price
            #     )

            try:
                # ======= SELL LOGIC =======
                if proportion < 0: # and proportion >= -1:
                    # if proportion < -1:
                    #     proportion = -1.
                    # SELL LOGIC (proportion < 0)
                    sell_fraction = min(abs(proportion), 1.0)
                    units_to_sell = position * sell_fraction
                    # units_to_sell = min(self.portfolio[asset], units_to_sell)
                    notional = units_to_sell * trade_price
                    fee = self.compute_transaction_fee(notional)
                    net_cash = notional - fee

                    if units_to_sell <= 0 or notional < self.min_trade_value or net_cash < 0 or position < units_to_sell or position <= 0.:
                        # logger.info(
                        #     f"Sell skipped for {asset}: amount={units_to_sell:.4f}, notional={notional:.2f}, fee={fee:.2f}, position={position:.4f}"
                        # )
                        # self.historical_trades.at[self.timestamp, f'{asset}_intended_notional_trade_value'] = notional
                        # self.historical_trades.at[self.timestamp, f'{asset}_Categorical_is_intended_notional_trade_value'] = 1
                        # self.update_asset_pnl(asset, current_price, timestamp)
                        self.live_pnl.process_trade(asset, 0, trade_price, "sell")
                        # self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)
                        for i, asset_name in enumerate(self.selected_assets):
                            if asset_name == asset:
                                blocked_counts[i] += 1
                        return # blocked_counts

                    # Update state
                    self.portfolio[asset] -= units_to_sell
                    self.portfolio['cash'] += net_cash
                    # self.balance += net_cash
                    self.total_transaction_fee += fee
                    self.traded += 1

                    # Record trade
                    self.historical_trades.at[self.timestamp, f'{asset}_trade_amount'] = -units_to_sell
                    self.historical_trades.at[self.timestamp, f'{asset}_cash_cost_revenue'] = net_cash
                    self.historical_trades.at[self.timestamp, f'{asset}_transaction_fee'] = fee
                    self.historical_trades.at[self.timestamp, f'{asset}_trade_value'] = notional
                    # self.historical_trades.at[self.timestamp, f'{asset}_Categorical_is_intended_notional_trade_value'] = 0

                    portfolio_sold_value += notional
                    total_transaction_fee += fee
                    portfolio_total_transactions_value += notional

                    # Update live_pnl if exists
                    if hasattr(self, 'live_pnl'):
                        self.live_pnl.process_trade(asset, -units_to_sell, trade_price, "sell")
                        # self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)

                    # Update tracking variables (BTC/ETH/SP500 quantities)
                    # if asset == 'Close':
                    #     self.btc_quantity = self.portfolio[asset]
                    #     self.btc_price = trade_price
                    # elif asset == 'ETH Close':
                    #     self.eth_quantity = self.portfolio[asset]
                    #     self.eth_price = trade_price
                    # elif asset == 'SP500 Close':
                    #     self.sp500_quantity = self.portfolio[asset]
                    #     self.sp500_price = trade_price

                    # return blocked_counts

                # ======= BUY LOGIC =======
                elif proportion > 0: # and proportion <= 1:
                    # if proportion == 1.:
                    #     fee = self.compute_transaction_fee(notional)
                    #     proportion = (cash - 1 - fee) / cash # -$1 fixed fee so that it fits the below check: cash < (intended_notional + fee)

                    # BUY LOGIC (proportion > 0)
                    intended_notional = cash * min(proportion, 1.0) # This setup have problem with the edge case where proportion/action is 1.
                    fee = self.compute_transaction_fee(intended_notional)
                    # intended_notional = min(self.balance, float(intended_notional + fee))

                    # if intended_notional < self.min_trade_value or cash < (intended_notional + fee): # Original.
                    if intended_notional < self.min_trade_value or self.balance < (intended_notional + fee) or intended_notional <= fee or float(intended_notional + fee) <= 0 or self.balance <= 0. or intended_notional <= 0.:
                        # logger.info(
                        #     f"Buy skipped for {asset}: trade_value={intended_notional:.2f}, fee={fee:.2f}, cash={cash:.2f}"
                        #     )
                        # self.historical_trades.at[self.timestamp, f'{asset}_intended_notional_trade_value'] = -intended_notional
                        # self.historical_trades.at[self.timestamp, f'{asset}_Categorical_is_intended_notional_trade_value'] = 1
                        self.live_pnl.process_trade(asset, 0, trade_price, "buy")
                        # self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)
                        for i, asset_name in enumerate(self.selected_assets):
                            if asset_name == asset:
                                blocked_counts[i] += 1
                        return # blocked_counts

                    units_bought = (intended_notional - fee) / trade_price
                    if units_bought <= 0:
                        # logger.info(
                        #     f"Buy skipped for {asset}: fee ({fee:.2f}) >= notional ({intended_notional:.2f})"
                        #     )
                        # self.historical_trades.at[self.timestamp, f'{asset}_intended_notional_trade_value'] = -intended_notional
                        # self.historical_trades.at[self.timestamp, f'{asset}_Categorical_is_intended_notional_trade_value'] = 1
                        self.live_pnl.process_trade(asset, 0, trade_price, "buy")
                        # self.get_unrealized_pnl_fifo(asset, trade_price)
                        self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)
                        for i, asset_name in enumerate(self.selected_assets):
                            if asset_name == asset:
                                blocked_counts[i] += 1
                        return # blocked_counts

                    self.portfolio['cash'] -= intended_notional
                    self.portfolio[asset] += units_bought
                    # self.balance -= intended_notional
                    self.total_transaction_fee += fee
                    self.traded += 1

                    # Record trade
                    self.historical_trades.at[self.timestamp, f'{asset}_trade_amount'] = units_bought
                    self.historical_trades.at[self.timestamp, f'{asset}_cash_cost_revenue'] = -intended_notional
                    self.historical_trades.at[self.timestamp, f'{asset}_transaction_fee'] = fee
                    self.historical_trades.at[self.timestamp, f'{asset}_trade_value'] = intended_notional
                    # self.historical_trades.at[self.timestamp, f'{asset}_Categorical_is_intended_notional_trade_value'] = 0

                    portfolio_bought_value += intended_notional
                    total_transaction_fee += fee
                    portfolio_total_transactions_value += intended_notional

                    # Update live_pnl if exists
                    if hasattr(self, 'live_pnl'):
                        self.live_pnl.process_trade(asset, units_bought, trade_price, "buy")
                        self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)

                    # Update tracking variables (BTC/ETH/SP500 quantities)
                    # if asset == 'Close':
                    #     self.btc_quantity = self.portfolio[asset]
                    #     self.btc_price = trade_price
                    # elif asset == 'ETH Close':
                    #     self.eth_quantity = self.portfolio[asset]
                    #     self.eth_price = trade_price
                    # elif asset == 'SP500 Close':
                    #     self.sp500_quantity = self.portfolio[asset]
                    #     self.sp500_price = trade_price

                    # return blocked_counts

                # NO TRADE
                # else:
                    # self.live_pnl.process_trade(asset, 0, trade_price, "buy")
                    # self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)
                    # for i, asset_name in enumerate(self.selected_assets):
                    #     if asset_name == asset:
                    #         blocked_counts[i] += 1
                    # return blocked_counts

                # ======= PnL (Optional, skip if not needed) =======
                try: # --- PnL block unchanged ---
                    if hasattr(self, 'live_pnl'):
                        # self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)
                        fifo_r, fifo_u, lifo_r, lifo_u = self.live_pnl.update_asset_pnl(asset, trade_price, self.timestamp)

                        total_fifo_pnl = self.live_pnl.get_total_pnl(asset, current_price=trade_price, method="FIFO")
                        total_lifo_pnl = self.live_pnl.get_total_pnl(asset, current_price=trade_price, method="LIFO")
                        self.historical_trades.at[self.timestamp, f"{asset}_LIFO_Total_PnL"] = float(total_lifo_pnl)
                        self.historical_trades.at[self.timestamp, f"{asset}_FIFO_Total_PnL"] = float(total_fifo_pnl)


                        # pnls = self.live_pnl.get_all_asset_pnls()
                        # fifo_r = self.live_pnl.get_realized_pnl_fifo(asset)
                        # fifo_u = self.live_pnl.get_unrealized_pnl_fifo(asset, current_price=trade_price)
                        # lifo_r = self.live_pnl.get_realized_pnl_lifo(asset)
                        # lifo_u = self.live_pnl.get_unrealized_pnl_lifo(asset, current_price=trade_price)

                        if fifo_r != 0.:
                            self.historical_trades.at[self.timestamp, f"FIFO_realized_{asset}_PnL"] = fifo_r
                        if fifo_r == 0. or pd.isna(fifo_r):
                            self.historical_trades.at[self.timestamp, f"FIFO_realized_{asset}_PnL"] = self.historical_trades.at[self.previous_timestamp, f"FIFO_realized_{asset}_PnL"]
                        if fifo_u != 0.:
                            self.historical_trades.at[self.timestamp, f"FIFO_unrealized_{asset}_PnL"] = fifo_u
                        if fifo_u == 0. or pd.isna(fifo_u):
                            previous_asset_quantity = float(self.historical_trades.at[self.previous_timestamp, f"portfolio_{asset}_owned"])
                            current_price = float(trade_price)
                            # self.historical_trades.at[self.timestamp, f"FIFO_unrealized_{asset}_PnL"] = self.historical_trades.at[self.previous_timestamp, f"FIFO_unrealized_{asset}_PnL"]
                            self.historical_trades.at[self.timestamp, f"FIFO_unrealized_{asset}_PnL"] = float(float(trade_price) * float(previous_asset_quantity))
                        if lifo_r != 0.:
                            self.historical_trades.at[self.timestamp, f"LIFO_realized_{asset}_PnL"] = lifo_r
                        if lifo_r == 0. or pd.isna(lifo_r):
                            self.historical_trades.at[self.timestamp, f"LIFO_realized_{asset}_PnL"] = self.historical_trades.at[self.previous_timestamp, f"LIFO_realized_{asset}_PnL"]
                        if lifo_u != 0.:
                            self.historical_trades.at[self.timestamp, f"LIFO_unrealized_{asset}_PnL"] = lifo_u
                        if lifo_u == 0. or pd.isna(lifo_u):
                            # self.historical_trades.at[self.timestamp, f"LIFO_unrealized_{asset}_PnL"] = self.historical_trades.at[self.previous_timestamp, f"LIFO_unrealized_{asset}_PnL"]
                            previous_asset_quantity = float(self.historical_trades.at[self.previous_timestamp, f"portfolio_{asset}_owned"])
                            current_price = float(trade_price)
                            self.historical_trades.at[self.timestamp, f"LIFO_unrealized_{asset}_PnL"] = float(float(trade_price) * float(previous_asset_quantity))

                except Exception as e:
                    logger.error("Error updating PnL for %s after trade: %s", asset, str(e))

            except Exception as e:
                logger.error("Error executing trade for asset %s: %s", asset, str(e))

            return blocked_counts

        # 1) Execute your trades, detect which assets got blocked: # For Dirchlet Prior of Blocked_trades.
        blocked_counts = np.zeros(self.n_assets, int)

        self.cash_trade_action = float(0.)
        self.cash_intended_trade_weight = float(0.)
        before_sell_cash = float(self.balance)
        # === 1. Execute all SELLS ===
        # for asset, proportion, hrp_proportion in sells:
        for asset, proportion in sells:
            trade_price = current_prices[asset]
            proportion = round(proportion, 4)
            # blocked_counts = process_trade(asset, proportion, trade_price, hrp_proportion=hrp_actions[i], blocked_counts=blocked_counts) # Original.
            process_trade(asset, proportion, trade_price, hrp_proportion=hrp_actions[i])

        self.cash_trade_action += float(float(self.balance) - float(before_sell_cash))

        before_buy_cash = float(self.balance)
        # === 2. Execute all BUYS ===
        # for asset, proportion, hrp_proportion in buys:
        for asset, proportion in buys:
            if float(self.balance) != 0.:
                proportion = float(float(before_buy_cash * proportion) / float(self.balance))
            # else:
            #     proportion = float(0.)
            trade_price = current_prices[asset]
            proportion = round(proportion, 4)
            # blocked_counts = process_trade(asset, proportion, trade_price, hrp_proportion=hrp_actions[i], blocked_counts=blocked_counts)
            process_trade(asset, proportion, trade_price, hrp_proportion=hrp_actions[i])

        self.cash_trade_action += float(before_buy_cash) - float(self.balance)
        if before_sell_cash != 0.:
            self.cash_intended_trade_weight = float(float(self.cash_trade_action) / float(self.previous_portfolio_value))
        else:
            self.cash_intended_trade_weight = float(0.)

        # Per-timestep summary
        self.historical_trades.at[self.timestamp, 'before_buy_cash'] = float(before_buy_cash)
        self.historical_trades.at[self.timestamp, 'cash_trade_action'] = float(self.cash_trade_action)
        self.historical_trades.at[self.timestamp, 'cash_intended_trade_weight'] = float(self.cash_intended_trade_weight)
        self.historical_trades.at[self.timestamp, 'portfolio_bought_value'] = float(portfolio_bought_value)
        self.historical_trades.at[self.timestamp, 'portfolio_sold_value'] = float(portfolio_sold_value)
        self.historical_trades.at[self.timestamp, 'total_transaction_fee'] = float(total_transaction_fee)
        self.historical_trades.at[self.timestamp, 'portfolio_total_transactions_value'] = float(portfolio_total_transactions_value)

        # logger.info(
        #     f"Trades @ {self.timestamp}: Bought={portfolio_bought_value:.2f}, Sold={portfolio_sold_value:.2f}, Fees={total_transaction_fee:.2f}"
        # )
        return blocked_counts



    # def execute_trade_dict(self, actions):
    def execute_trade_dict(self, asset, proportion, trade_price):
        # current_prices = self.get_current_prices()
        # timestamp = self.data.iloc[self.current_step]['timestamp']
        
        # for asset, proportion in actions.items():
        # trade_price = current_prices[asset]

        # Selling logic
        # if proportion < 0 and proportion >= -1 and self.hrp_portfolio[asset] * trade_price * proportion >= 10.:
        if proportion < 0 and proportion >= -1: # and float(abs(proportion) * self.hrp_portfolio[asset] * trade_price) >= 10.:
            # abs(action * self.hrp_portfolio[asset] * current_prices[asset])/Intended Sale of assets has to be >= 10.:
            trade_price = trade_price * (1 - self.transaction_fee)
            sell_amount = self.hrp_portfolio[asset] * abs(proportion)
            # desired_goals:
            # achieved_goals:
            # self.hrp_desired_goals.append(-sell_amount)
            if self.hrp_portfolio[asset] - sell_amount >= 0:
                self.hrp_portfolio[asset] -= sell_amount
            else:
                # sell_amount = self.hrp_portfolio[asset]
                self.hrp_portfolio[asset] = 0

            self.hrp_portfolio['cash'] += sell_amount * trade_price
            # self.hrp_balance += sell_amount * trade_price
            # self.hrp_portfolio[asset] -= sell_amount
            cash_revenue = sell_amount * trade_price 
            self.historical_trades.at[self.timestamp, f'hrp_{asset}_trade_amount'] = -float(sell_amount)
            self.historical_trades.at[self.timestamp, f'hrp_{asset}_cash_cost_revenue'] = float(cash_revenue)
            # self.historical_trades.at[self.timestamp, f'hrp_portfolio_{asset}_trade_value'] = float(

            if asset == 'Close':
                self.hrp_btc_quantity = self.hrp_portfolio[asset]
                self.hrp_btc_price = trade_price # current_prices[asset]
                # self.hrp_btc_value = self.hrp_btc_quantity * self.hrp_btc_price
            elif asset == 'ETH Close':
                self.hrp_eth_quantity = self.hrp_portfolio[asset]
                self.hrp_eth_price = trade_price # current_prices[asset] 
                # self.hrp_eth_value = self.hrp_eth_quantity * self.hrp_eth_price
            elif asset == 'SP500 Close':
                self.hrp_sp500_quantity = self.hrp_portfolio[asset]
                self.hrp_sp500_price = trade_price # current_prices[asset] 
                # self.hrp_sp500_value = self.hrp_sp500_quantity * self.hrp_sp500_price

            amount = -sell_amount
            # self.record_trade(
            #     asset=asset, volume=amount, trade_price=trade_price, strategy='self'
            #     )

        elif proportion > 0 and proportion <= 1: # and self.hrp_balance > 10. and self.hrp_portfolio['cash'] > 10. and abs(proportion) * self.hrp_balance >= 10.:
            # abs(action * self.hrp_balance) / Intended aciton have to be >= 10.:
            trade_price = trade_price * (1 + self.transaction_fee)
            buy_amount = self.hrp_portfolio['cash'] * proportion / trade_price
            cash_cost = buy_amount * trade_price
            # desired_goal
            self.historical_trades.at[self.timestamp, f'hrp_{asset}_trade_amount'] = float(buy_amount)
            self.historical_trades.at[self.timestamp, f'hrp_{asset}_cash_cost_revenue'] = -float(cash_cost)
            # achieved_goal
            if self.hrp_portfolio['cash'] - cash_cost >= 0:
                self.hrp_portfolio['cash'] -= buy_amount * trade_price
                # self.hrp_balance -= buy_amount * trade_price
            else:
                # temp fix:
                buy_amount = self.portfolio['cash']
                self.hrp_portfolio['cash'] = 0
                # self.hrp_balance = 0
            self.hrp_portfolio[asset] += buy_amount

            if asset == 'Close':
                self.hrp_btc_quantity = self.hrp_portfolio[asset]
                self.hrp_btc_price = trade_price # current_prices[asset]
                # self.hrp_btc_value = self.hrp_btc_quantity * self.hrp_btc_price
            elif asset == 'ETH Close':
                self.hrp_eth_quantity = self.hrp_portfolio[asset]
                self.hrp_eth_price = trade_price # current_prices[asset]
                # self.hrp_eth_value = self.hrp_eth_quantity * self.hrp_eth_price
            elif asset == 'SP500 Close':
                self.hrp_sp500_quantity = self.hrp_portfolio[asset]
                self.hrp_sp500_price = trade_price # current_prices[asset]
                # self.hrp_sp500_value = self.hrp_sp500_quantity * self.hrp_sp500_price
            # self.record_trade(
            #     asset=asset, volume=cash_cost, trade_price=trade_price, strategy='self'
            #     )
        # else:
        #     print(
        #         f'asset is: {asset} current_prices are: {current_prices} timestamp is: {self.timestamp} current_prices[asset] is: {current_prices[asset]} trade_price {trade_price} proportionis: {proportion}'
        #         )


    def calculate_parametric_var(self, confidence_level=0.95):
        # values = np.asarray(self.portfolio_values)
        if len(self.portfolio_values) < 3:
            return 0.0
        # returns = values[1:] / values[:-1] - 1
        if len(self.returns) < 2:
            return 0.0
        # mu = np.mean(returns)
        mu = np.mean(self.returns)
        sigma = np.std(self.returns, ddof=1)
        # from scipy.stats import norm
        z = norm.ppf(1 - confidence_level)
        var = mu + z * sigma
        return var

    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown from historical portfolio values.
        
        Returns:
            float: The maximum drawdown value (as a negative decimal, e.g., -0.25 for -25%)
        """
        # Convert to numpy array for efficient computation
        values = np.array(self.portfolio_values)
        if len(values) < 2:
            return 0.0  # No drawdown if fewer than 2 values

        # Compute the running maximum up to each point
        running_max = np.maximum.accumulate(values)
        
        # Compute drawdowns at each step
        drawdowns = (values - running_max) / running_max # Original.
        # drawdowns = values - running_max
        
        # Max drawdown is the minimum value (most negative drawdown)
        max_drawdown = np.min(drawdowns)
        
        return max_drawdown


    def calculate_cumulative_reward(self, cumulative_return):#, window_size=10):
        window_size=len(self.cumulative_history)
        if window_size==0:
            volatility = 1e-5
            recent_returns = self.cumulative_returns_history.item()
        elif window_size==1:
            volatility = 1e-5
            recent_returns = self.cumulative_returns_history[-window_size:]
        else:
            # self.cumulative_returns_history stores the history of cumulative returns
            recent_returns = self.cumulative_returns_history[-window_size:]
            volatility = np.std(recent_returns)
        
        # Avoid division by zero #self.cumulative_returns_history[-1]-previous
        if volatility == 0 or volatility is np.isnan(volatility) or volatility is np.isinf(volatility):
            volatility = 1e-5
        
        # Sharpe Ratio-like adjustment
        risk_adjusted_return = cumulative_return / volatility
        return risk_adjusted_return # np.tanh(risk_adjusted_return)  # Using tanh for normalization

    def sigmoid(self, x):
        """Compute the sigmoid function."""
        #return 1 / (1 + np.exp(-x))
        return np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda : self])
        obs = e.reset()
        return e, obs

    

import pickle

def save_scaler_pickle(scaler: RunningScaler, filename: str = "running_scaler.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filename}")

def load_scaler_pickle(filename: str = "running_scaler.pkl") -> RunningScaler:
    with open(filename, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filename}")
    return scaler

def save_scaler_npz(scaler: RunningScaler, filename: str = "running_scaler.npz"):
    np.savez(filename, mean=scaler.mean, var=scaler.var, count=scaler.count)
    print(f"Scaler stats saved to {filename}")

def load_scaler_npz(filename: str = "running_scaler.npz") -> RunningScaler:
    data = np.load(filename)
    scaler = RunningScaler(shape=data["mean"].shape[0])
    scaler.mean = data["mean"]
    scaler.var = data["var"]
    scaler.count = data["count"]
    print(f"Scaler stats loaded from {filename}")
    return scaler


from PCAPreprocessor import PCAPreprocessor
from pathlib import Path

def neutralization(data, start_step_train, train_ratio=0.7):
    # Define the directory to save models
    model_save_dir = '/home/jashha/' # '/home/jashha/YC/Models/' # '/home/jashha/'
    os.makedirs(model_save_dir, exist_ok=True)

    # Load and scale data
    price_data = pd.read_parquet('/home/jashha/merged_data.parquet').replace([-np.inf, np.inf], np.nan).fillna(value=0) # Original. # updated_features_df.parquet

    data = pd.read_parquet('/home/jashha/merged_data.parquet').fillna(0).replace([-np.inf, np.inf], np.nan).fillna(value=0) # Original.

    print(data)
    print(data.head())
    print(data.tail())
 
    price_data.set_index('timestamp', inplace=True)
    price_data.reset_index('timestamp', inplace=True) # , drop=True)
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp']) # , unit='d') #'s')
    data['timestamp'] = pd.to_datetime(data['timestamp']) # , unit='d')


    data = data[
            data.timestamp.between(
                pd.to_datetime('20200101'), pd.to_datetime('20230617')
                # pd.to_datetime('20190101'), pd.to_datetime('20230617')
                )
            ]
    price_data = price_data[
            price_data.timestamp.between(
                pd.to_datetime('20200101'), pd.to_datetime('20230617')
                # pd.to_datetime('20190101'), pd.to_datetime('20230617')
                )
            ]

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.drop(columns=['Unnamed: 0'], errors='ignore')
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

    print(data)
    print(data.head())
    print(data.tail())

    
    split_idx = start_step_train + int((len(data) - start_step_train) * train_ratio)
    test_size = int(len(data) - split_idx) / len(data)

   

    selected_assets = ['Close', 'ETH Close', 'SP500 Close']
    columns = ['timestamp']
    columns += selected_assets
    columns += ['Adj Close', 'Open', 'High', 'Low', 'Volume']
    columns += ['ETH Adj Close', 'ETH Open', 'ETH High', 'ETH Low', 'ETH Volume']
    columns += ['SP500 Adj Close', 'SP500 Open', 'SP500 High', 'SP500 Low', 'SP500 Volume']

    price_data = price_data[columns]
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    columns.pop(0)
    print(f'columns: {columns}')
    # stop
    data[columns] = price_data[columns]
    print(data)
    print(data.head())
    print(data.tail())

    price_data.set_index('timestamp', inplace=True)
    data.set_index('timestamp', inplace=True, drop=True)
    # Split data into train and test sets
    train_data, test_data_ = train_test_split(
        data, test_size=float(test_size), shuffle=False
        )

    train_price_data, test_price_data_ = train_test_split(
        price_data, test_size=float(test_size), shuffle=False
        )
    # Split the test into validation and test data at ratio of 0.67 : 0.33 so that the train is 70% and the validation(test) is 20% and the validation is 10%.
    validation_data, test_data = train_test_split(
        test_data_, test_size=float(0.6666667), shuffle=False
        )
    validation_price_data, test_price_data = train_test_split(
        test_price_data_, test_size=float(0.6666667), shuffle=False
        )

    
    
    output_dir = './pca_pipeline'

    # Initialize and fit
    preprocessor = PCAPreprocessor(
        n_components=0.95,
        save_dir='./pca_pipeline',
    )
    features_col = [col for col in train_data.columns if col != 'timestamp']
    preprocessor.fit(train_data, features_col)

    # Transform data
    train_transformed = preprocessor.transform(train_data, features_col)
    validation_transformed = preprocessor.transform(validation_data, features_col)

    # Persist transformed CSVs
    train_out = Path(output_dir) / 'train_pca.csv'
    validation_out = Path(output_dir) / 'validation_pca.csv'
    train_out_parquet = Path(output_dir) / 'train_pca.parquet'
    validation_out_parquet = Path(output_dir) / 'validation_pca.parquet'
    train_transformed.to_csv(train_out)
    validation_transformed.to_csv(validation_out)
    train_transformed.to_parquet(train_out_parquet)
    validation_transformed.to_parquet(validation_out_parquet)

    
    del train_data
    del test_data
    del validation_data

    train_data = deepcopy(train_transformed)
    validation_data = deepcopy(validation_transformed)

    # Scale the data
    train_scaler = MinMaxScaler(feature_range=(0, 1)) # RobustScaler() # StandardScaler() # MinMaxScaler(feature_range=(0, 1))
    train_scaler.fit(train_data) # .iloc[:start_step_train].values)
    train_data_ = train_scaler.transform(train_data)
    validation_data_ = train_scaler.transform(validation_data)

    train_data_scaled = pd.DataFrame(
        train_data_,
        index=train_data.index,
        columns=train_data.columns
        )

    validation_data_scaled = pd.DataFrame(
        validation_data_,
        index=validation_data.index,
        columns=validation_data.columns
        )
    print(f'||| train_data_scaled: {train_data_scaled} ||| validation_data_scaled: {validation_data_scaled} ||| max: {train_data_scaled.max()} ||| max: {validation_data_scaled.max()} ||| min: {train_data_scaled.min()} ||| min: {validation_data_scaled.min()} |||')

    train_data.reset_index('timestamp', inplace=True, drop=False)
    validation_data.reset_index('timestamp', inplace=True, drop=False)
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    validation_data['timestamp'] = pd.to_datetime(validation_data['timestamp'])

    test_scaler = deepcopy(train_scaler)
    return train_data, validation_data, train_price_data, validation_price_data, train_scaler, test_scaler  # running_scaler

























    




    
            





