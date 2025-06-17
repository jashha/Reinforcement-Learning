
def test(agent, num_episodes): # Works.
    """
        Test the agent in the test environment using deterministic actions.
    """
    # eval_mode(agent)
    test_episode_rewards = []
    test_episode_lengths = []
    test_episode_portfolio_values = []
    test_episode_hrp_portfolio_values = []
    test_sharpe_ratios = []
    num_test_episodes = num_episodes
    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = agent.test_env.reset(), False, 0, 0
        while not (d or (ep_len == agent.test_env.max_ep_len)):
            # a = agent.get_action(o=o, deterministic=True)
            a = agent.ac.act(obs=torch.FloatTensor(o).to(device, non_blocking=True))
            o, r, d, _ = agent.test_env.step(a)
            # test_env.render()
            ep_ret += r
            ep_len += 1
        agent.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        test_episode_rewards.append(ep_ret)
        test_episode_lengths.append(ep_len)
        # test_episode_hrp_portfolio_values.extend(np.mean(test_env.hrp_portfolio_values))
        test_episode_portfolio_values.append(np.mean(agent.test_env.portfolio_values))
        # test_episode_hrp_portfolio_values.append(np.mean([a - b for a, b in zip(agent.test_env.portfolio_values, agent.test_env.hrp_portfolio_values)]))
        test_sharpe_ratios.append(np.mean(agent.test_env.historical_trades['sharpe']))

    # Log final test statistics
    avg_test_reward = np.mean(test_episode_rewards)
    max_test_reward = np.max(test_episode_rewards)
    # logging.info(f'Avg Test Reward: {avg_test_reward}, Max Test Reward: {max_test_reward}')
    avg_test_lengths = np.mean(test_episode_lengths)
    max_test_lengths = np.max(test_episode_lengths)
    # logging.info(f'Avg Test Lengths: {avg_test_lengths}, Max Test Lengths: {max_test_lengths}')
    # return avg_test_reward, max_test_reward, avg_test_lengths, max_test_lengths, test_episode_rewards, test_episode_lengths, test_episode_portfolio_values, test_episode_hrp_portfolio_values, agent.test_env.historical_trades # test_csv # test_sharpe_ratio
    return avg_test_reward, max_test_reward, avg_test_lengths, max_test_lengths, test_episode_rewards, test_episode_lengths, test_episode_portfolio_values, test_sharpe_ratios, agent.test_env.historical_trades




def train_ppo(
    ppo_agent, num_episodes_train,
    num_episodes_test, is_hyper_tune=False
    ):
    start_step = ppo_agent.env.start_step
    end_step = ppo_agent.env.end_step
    max_step = end_step - 1 - start_step

    # max_ep_len (int): Maximum length of trajectory / episode / rollout.
    max_ep_len = int(end_step - 1 - start_step)
    # steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
    #         for the agent and the environment in each epoch.
    steps_per_epoch = max_step # - 1 # ppo_agent.env.max_step - 1steps_per_epoch
    max_ep_len = ppo_agent.env.max_ep_len + 1

    # logger_kwargs = setup_logger_kwargs(exp_name='ppo2_with_autoencoder_decoder', seed=2024, data_dir='/home/jashha/', datestamp=False)
    # logger_kwargs = setup_logger_kwargs(
    #     exp_name='PPO_MLP', seed=42, data_dir='/home/jashha/', datestamp=False
    #     )

    # logger_kwargs = setup_logger_kwargs(
    #     exp_name='ppo_continuous', seed=2024, data_dir='/home/jashha/', datestamp=False
    #     )

    model_dir  = "./results/training_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir, "ppo_actor_critic.pth"
        )

    # if not use_2d:
    #     state_dim = len(ppo_agent.env.get_state())
    # else:
    state_dim = ppo_agent.env.observation_space.shape[0]
    # print(f'state_dim: {state_dim}')
    # stop
    action_dim = ppo_agent.env.action_space.shape[0]
    # action_dim = len(selected_assets)
    # print(f'action_dim: {action_dim}')
    # print(f'env: {ppo_agent.env} ||| ppo_agent.test_env: {ppo_agent.test_env}')
    # print(f'env: {ppo_agent.env.render()} ||| test_env: {ppo_agent.test_env.render()}')
    # stop

    # max_step = ppo_agent.env.end_step - 1 - ppo_agent.env.start_step
    start_step = ppo_agent.env.start_step
    end_step = ppo_agent.env.end_step
    # max_step = end_step - 1 - start_step
    max_step = ppo_agent.env.max_ep_len
    # max_ep_len (int): Maximum length of trajectory / episode / rollout.
    # max_ep_len = int(end_step - 1 - start_step)
    max_ep_len = ppo_agent.env.max_ep_len

    ########### Training loop ###########
    print('########### ENTERING TRAINING LOOP ##################')
    n_episodes = num_episodes_train
    
    # Evulation stats:
    episodes_rewards = []
    final_rewards = []
    best_episodes_rewards = []
    average_episodes_rewards = []
    std_episodes_rewards = []
    sum_episodes_rewards = []
    median_episodes_rewards = []
    train_sharpe_ratios = [] # | ExplainedVariance |
    test_sharpe_ratios = [] 
    # ppo_agent.ac.
    # steps_per_epoch = ppo_agent.env.end_step - 1 - ppo_agent.env.start_step
    steps_per_epoch = ppo_agent.env.max_ep_len + 1

    steps_per_epoch = ppo_agent.env.max_ep_len
    epochs = num_episodes_train

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = ppo_agent.env.reset(), 0, 0
    local_steps_per_epoch = int(steps_per_epoch / num_procs()) # Original.
    # local_steps_per_epoch = int(steps_per_epoch) # * num_procs())

    episodes_rewards = []
    episode_rewards = []
    episode_lengths = []
    results = {}


    train_episode_rewards = []
    train_episode_lengths = []
    test_episode_rewards = []
    test_episode_lengths = []
    train_episode_portfolio_values = []
    test_episode_portfolio_values = []
    train_episode_hrp_portfolio_values= []
    test_episode_hrp_portfolio_values = []

    '''avg_test_reward, max_test_reward, avg_test_lengths, max_test_lengths, test_episode_reward, test_episode_length, test_episode_portfolio_value, test_episode_hrp_portfolio_value = test(agent=ppo_agent, num_episodes=num_episodes_test)
    test_episode_rewards.extend(test_episode_reward)
    test_episode_lengths.extend(test_episode_length)
    test_episode_portfolio_values.extend(test_episode_portfolio_value)
    test_episode_hrp_portfolio_values.extend(
        test_episode_hrp_portfolio_value
        )'''

    progress_bar = tqdm(
        total=total_steps, desc="PPO Training Progress"
        )

    with tqdm(total=num_episodes_train, desc="PPO Training Progress") as pbar:
        '''try:
            # best_episode_reward = float('-inf')
            best_episode_reward = np.load("/home/jashha/Episode_rewards.npy") # FINALEEALE_episode_rewards.npy

        except Exception as e:
            best_episode_reward = float('-inf')
            print(f"Error loading rewards: {e}")'''
        # train_mode(ppo_agent)
        episode_len = 0
        actions = []
        episode_reward = 0
        count = 0

        for epoch in range(num_episodes_train): # total_steps): # num_episodes_train):
            for t in range(ppo_agent.steps_per_epoch): # local_steps_per_epoch):
                a, v, logp = ppo_agent.ac.step(
                    o
                    # torch.as_tensor(o, dtype=torch.float32).to(device, non_blocking=True) # Original.
                    ) # .unsqueeze(0))
                 
                next_o, r, d, _ = ppo_agent.env.step(a) # tanh_a) # .squeeze() # np.tanh(a))
                # print(f'||| a_raw: {a} ||| tanh_a: {tanh_a} |||')
                # next_o, r, d, _ = ppo_agent.env.step(w)
                ep_ret += r
                ep_len += 1

                # save and log
                # buf.store(o, a, r, v, logp)
                # logger.store(VVals=v)
                ppo_agent.remember(
                    state=o,
                    action=a, # a, # raw_action_sample.cpu().numpy(),
                    reward=r,
                    new_state=next_o,
                    done=d,
                    value=v,
                    log_probs=logp,
                    # action_weights=w,
                    )
                ppo_agent.logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o
                # state = next_state

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                # epoch_ended = t==ppo_agent.local_steps_per_epoch-1 # Original.
                epoch_ended = t==ppo_agent.steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = ppo_agent.ac.step(
                            o
                            # torch.as_tensor(o, dtype=torch.float32).to(device, non_blocking=True) # Original.
                            ) # .unsqueeze(0))
                    else:
                        v = 0
                    print(f"||| finishing_path(v)'s v value: {v} ||| reward, r is: {r}")
                    ppo_agent.replay_buffer.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        ppo_agent.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        # print(f'EpRet={episode_reward_t}, EpLen={ep_len}')
                        train_episode_rewards.append(ep_ret)
                        train_episode_lengths.append(ep_len)
                        train_episode_portfolio_values.append(np.mean(ppo_agent.env.portfolio_values))
                        train_sharpe_ratios.append(np.mean(ppo_agent.env.historical_trades['sharpe']))
                        # train_episode_hrp_portfolio_values.append(
                        #     np.mean(
                        #         [
                        #             a - b for a, b in zip(
                        #                 ppo_agent.env.portfolio_values, ppo_agent.env.hrp_portfolio_values
                        #                 )
                        #             ]
                        #         )
                        #     )
                        episode_lengths.append(ep_len)
                    o, ep_ret, ep_len = ppo_agent.env.reset(), 0, 0
                episode_rewards.append(ep_ret)

                # Update progress bar
                progress_bar.set_postfix({ # tqdm(total=total_steps, desc="PPO Training Progress")
                    "Epoch": epoch, # "/": num_episodes_train,
                    "Running Reward:": r,
                    "Running Total Reward:": ep_ret,
                    "Average Train Episode Reward": np.mean(train_episode_rewards) if train_episode_rewards else 0.0,
                    # "Average Train Episode Length": np.mean(train_episode_lengths) if train_episode_lengths else 0.0,
                    "Average Test Episode Reward": np.mean(test_episode_rewards) if test_episode_rewards else 0.0,
                    # "Max Average Episode Reward": max_test_reward if max_test_reward else 0.0,
                    # "Test Average Episode Length": np.mean(test_episode_lengths) if test_episode_lengths else 0.0,
                    # "Test Max Episode Length": max_test_lengths if max_test_lengths else 0.0,
                    "step": t,"/": steps_per_epoch,
                    # "entropy": entropy if entropy else 0.0
                })
                progress_bar.update(1)

            # Perform PPO update!
            # update()
            ppo_agent.update(step=epoch) # t)

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     ppo_agent.logger.save_state({'env': env}, None)
            #     print(f'env: {ppo_agent.env.render()}') # , None)
            try:
                # if (epoch % ppo_agent.save_freq == 0) or (epoch == epochs-1):
                if (epoch % ppo_agent.save_freq == 0):
                    if epoch != 0:
                        print(f'||| epoch: {epoch} ||| ppo_agent.save_freq: {ppo_agent.save_freq} |||')
                        if not is_hyper_tune:
                            if os.path.exists(model_path):
                                print(f'||| epoch: {epoch} ||| ppo_agent.save_freq: {ppo_agent.save_freq} |||')
                                torch.save(
                                    ppo_agent.ac.state_dict(), model_path
                                    )
                                print(
                                    "Saved model to", model_path
                                    )
                            '''else:
                                # print("Training final model with best hyper-parameters…")
                                # train_ppo(
                                #     final_agent, args.train_episodes, 0
                                #     )
                                torch.save( 
                                    ppo_agent.ac.state_dict(), model_path
                                    )
                                print(
                                    "Saved model to", model_path
                                    )'''
                    '''# === Example usage in training script ===
                    # During training (e.g., every N steps or on exit):
                    state = {
                        # 'step'            : t, # current_step,
                        'actor_state'     : ppo_agent.ac.pi.state_dict(),
                        'value_state'    : ppo_agent.ac.v.state_dict(),
                        'actor_optimizer_state' : ppo_agent.pi_optimizer.state_dict(),
                        'value_optimizer_state' : ppo_agent.v_optimizer.state_dict(),
                        # 'scheduler_state' : scheduler.state_dict() if scheduler else None,
                        # 'rng_state'       : torch.get_rng_state(),
                    }
                    save_path = save_checkpoint(state, checkpoint_dir="./checkpoints")'''
                    # torch.save(
                    #     ppo_agent.ac.state_dict(), model_path
                    #     )
                    # print(
                    #     "Saved final model to", model_path
                    #     )
                    # print(f"[Info] Checkpoint saved to {model_path}")
                    # print(f"Model saved at epoch {epoch}")

            except Exception as e:
                print(f"Error loading rewards: {e}")

            # Test the agent periodically
            # avg_test_reward, max_test_reward, avg_test_lengths, max_test_lengths, test_episode_reward, test_episode_length, test_episode_portfolio_value, test_episode_hrp_portfolio_value, test_csv = test(agent=ppo_agent, num_episodes=num_episodes_test)
            avg_test_reward, max_test_reward, avg_test_lengths, max_test_lengths, test_episode_reward, test_episode_length, test_episode_portfolio_value, test_sharpe_ratio, test_csv = test(agent=ppo_agent, num_episodes=num_episodes_test)
            # test_sharpe_ratios
            # train_mode(ppo_agent)

            test_episode_rewards.extend(test_episode_reward)
            test_episode_lengths.extend(test_episode_length)
            test_episode_portfolio_values.extend(test_episode_portfolio_value)
            test_sharpe_ratios.extend(test_sharpe_ratio)
            # test_episode_hrp_portfolio_values.extend(
            #     test_episode_hrp_portfolio_value
            #     )

            try:
                # plot_results(episode_rewards=train_episode_rewards, episode_lengths=train_episode_lengths, test_episode_rewards=test_episode_rewards, test_episode_lengths=test_episode_lengths)
                # plot_results(episode_rewards=mpi_avg(train_episode_rewards), episode_lengths=train_episode_lengths, test_episode_rewards=mpi_avg(test_episode_rewards), test_episode_lengths=test_episode_lengths)
                # train_episode_portfolio_values = mpi_avg(train_episode_portfolio_values)
                plot_training_results(
                    results=train_episode_portfolio_values, metric="Training Portfolio Value"
                    )
                # train_episode_rewards = mpi_avg(train_episode_rewards)
                # test_episode_rewards = mpi_avg(test_episode_rewards)
                plot_results(
                    episode_rewards=train_episode_rewards, episode_lengths=train_episode_lengths,
                    test_episode_rewards=test_episode_rewards, test_episode_lengths=test_episode_lengths
                    )
                # test_episode_portfolio_values = mpi_avg(test_episode_portfolio_values)
                plot_training_results(
                    results=test_episode_portfolio_values, metric="Testing Portfolio Value"
                    )
                plot_training_results( 
                    results=train_sharpe_ratios, metric="Train Sharpe Ratios"
                    )
                plot_training_results(
                    results=test_sharpe_ratios, metric="Test Sharpe Ratios"
                    )
                '''plot_training_results(
                    results=train_episode_hrp_portfolio_values, metric="Train Portfolio Value over HRP Portfolio Value"
                    )
                plot_training_results(
                    results=test_episode_hrp_portfolio_values, metric="Test Portfolio Value over HRP Portfolio Value"
                    )'''

            except Exception as e:
                # logger.error("Error executing trade for asset %s: %s", asset, str(e))
                # logger.error("Error plotting", str(e))
                print("Error plotting", str(e))
            
            # Log info about epoch
            ppo_agent.logger.log_tabular('Epoch', epoch)
            ppo_agent.logger.log_tabular('EpRet', with_min_and_max=True)
            ppo_agent.logger.log_tabular('EpLen', average_only=True)
            ppo_agent.logger.log_tabular('VVals', with_min_and_max=True)
            ppo_agent.logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            ppo_agent.logger.log_tabular('LossPi', average_only=True)
            ppo_agent.logger.log_tabular('LossV', average_only=True)
            ppo_agent.logger.log_tabular('DeltaLossPi', average_only=True)
            ppo_agent.logger.log_tabular('DeltaLossV', average_only=True)
            ppo_agent.logger.log_tabular('Entropy', average_only=True)
            ppo_agent.logger.log_tabular('KL', average_only=True)
            ppo_agent.logger.log_tabular('ClipFrac', average_only=True)
            ppo_agent.logger.log_tabular('StopIter', average_only=True)
            ppo_agent.logger.log_tabular('ExplainedVariance', average_only=True)
            ppo_agent.logger.log_tabular('Time', time.time()-start_time)
            ppo_agent.logger.dump_tabular()
            # ppo_agent.env.render()
            # pbar.update(1)
    
        '''# Update progress bar
        pbar.set_postfix({
            "Epoch": epoch, "/": num_episodes_train,
            "Train Episode Reward": np.mean(train_episode_rewards) if train_episode_rewards else 0.0,
            "Train Episode Length": np.mean(train_episode_lengths) if train_episode_lengths else 0.0,
            "Test Average Episode Reward": np.mean(test_episode_rewards) if test_episode_rewards else 0.0,
            # "Max Average Episode Reward": max_test_reward if max_test_reward else 0.0,
            "Test Average Episode Length": np.mean(test_episode_lengths) if test_episode_lengths else 0.0,
            # "Test Max Episode Length": max_test_lengths if max_test_lengths else 0.0,
            # "Steps": t
            "step": t,"/": local_steps_per_epoch,
            # "entropy": entropy if entropy else 0.0
        })
        pbar.update(1)'''

    # Close the progress bar
    progress_bar.close()

    # except Exception as e:
    #     print(f"Error loading rewards: {e}")

    # Save model
    # if (epoch % save_freq == 0) or (epoch == epochs-1):
    #     ppo_agent.logger.save_state({'env': env}, None)
    #     print(f'env: {ppo_agent.env.render()}') # , None)
    try:
        # if (epoch % ppo_agent.save_freq == 0) or (epoch == epochs-1) and epoch != 0:
        if (epoch % ppo_agent.save_freq == 0) and epoch != 0:
            print(f'||| epoch: {epoch} ||| ppo_agent.save_freq: {ppo_agent.save_freq} |||')
            
            if os.path.exists(model_path):
                print(f'||| epoch: {epoch} ||| ppo_agent.save_freq: {ppo_agent.save_freq} |||')
                torch.save(
                    ppo_agent.ac.state_dict(), model_path
                    )
                print(
                    "Saved model to", model_path
                    )
            else:
                # print("Training final model with best hyper-parameters…")
                # train_ppo(
                #     final_agent, args.train_episodes, 0
                #     )
                torch.save(
                    ppo_agent.ac.state_dict(), model_path
                    )
                print(
                    "Saved model to", model_path
                    )

            '''# === Example usage in training script ===
            # During training (e.g., every N steps or on exit):
            state = {
                # 'step'            : t, # current_step,
                'actor_state'     : ppo_agent.ac.pi.state_dict(),
                'value_state'    : ppo_agent.ac.v.state_dict(),
                'actor_optimizer_state' : ppo_agent.pi_optimizer.state_dict(),
                'value_optimizer_state' : ppo_agent.v_optimizer.state_dict(),
                # 'scheduler_state' : scheduler.state_dict() if scheduler else None,
                # 'rng_state'       : torch.get_rng_state(),
            }
            save_path = save_checkpoint(state, checkpoint_dir="./checkpoints")
            print(f"[Info] Checkpoint saved to {save_path}")
            print(f"Model saved at epoch {epoch}")'''

    except Exception as e:
        print(f"Error saving model: {e}")

    # Close the progress bar
    # progress_bar.close()
    # stop
    # test_csv = ppo_agent.test_env.historical_trades
    test_csv.to_csv(
        f'ppo_test_historical_trades.csv'
        )
    return avg_test_reward, max_test_reward, avg_test_lengths, max_test_lengths, test_episode_rewards, test_episode_lengths, test_episode_portfolio_values, test_episode_hrp_portfolio_values, test_csv








def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.LazyLinear(sizes[j+1], bias=True), act()]
        # layers += [BayesianLinear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def policy_mlp(sizes, activation, output_activation=nn.Identity, use_noisy=False):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation # Original.
        # act = activation if j < len(sizes)-2 else nn.Tanh
        # if act is not output_activation:
        if use_noisy:
            layers += [NoisyLazyLinear(sizes[j+1], bias=True), act()]
        else:
            layers += [nn.LazyLinear(sizes[j+1], bias=True), act()]
        # else:
        #     layers += [NoisyLazyLinear(sizes[j+1], bias=True), nn.Softmax(dim=-1)]
        # layers += [BayesianLinear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def get_hidden_dims(n_layers, hidden_sizes):
    return [hidden_sizes[0] if (i % 2 == 0) else hidden_sizes[-1] for i in range(n_layers)]



from sac_env_bak import TanhNormal

class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, n_layers, activation, use_action_norm=False):
        super().__init__()
        self.use_action_norm = use_action_norm
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32) # Original.
        # log_std = 0.70 * np.ones(act_dim, dtype=np.float32)
        # print(f'||| log_std: {log_std} ||| log_std1: {log_std1} ||| std: {log_std.exp()} ||| std1: {log_std1.exp()}')
        # stop
        hidden_dims = get_hidden_dims(n_layers=n_layers, hidden_sizes=hidden_sizes) # hidden1_dim, hidden2_dim)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)) # Original.
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
        self.mu_net = policy_mlp([obs_dim] + hidden_dims + [act_dim], activation) # , use_noisy=True)
        # self.mu_net = SquashNormalModulemlp(sizes=[obs_dim] + hidden_dims + [act_dim], activation=activation, output_activation=nn.Identity)
        # self.mu_net = policy_mlp([obs_dim] + list(hidden_sizes) + list(hidden_sizes) + [act_dim], activation) # Original.
        # self.mu_net = build_mlp(input_dim=obs_dim, output_dim=act_dim, n_layers=n_layers, hidden1_dim=hidden_sizes[0], hidden2_dim=hidden_sizes[-1], activation=activation, output_activation=nn.Identity)
        self.to(device, non_blocking=True)

    def _distribution(self, obs):
        mu = self.mu_net(torch.as_tensor(obs).to(device, non_blocking=True))
        # mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        if self.use_action_norm:
            return Normal(mu, std) # Original.
        else:
            return TanhNormal(loc=mu, scale=std) # Original.

    def _log_prob_from_distribution(self, pi, act):
        if self.use_action_norm:
            return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Normal distribution
        else:
            return pi.log_prob(act) # Original. # Last axis sum NOT needed for TanhNormal distribution



class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, n_layers, activation):
        super().__init__()
        hidden_dims = get_hidden_dims(n_layers=n_layers, hidden_sizes=hidden_sizes)
        self.v_net = mlp([obs_dim] + hidden_dims + [1], activation)
        # self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation) # Original.
        # self.v_net = build_mlp(input_dim=obs_dim, output_dim=1, n_layers=n_layers, hidden1_dim=hidden_sizes[0], hidden2_dim=hidden_sizes[-1], activation=activation, output_activation=nn.Identity)
        # NoisyLazyLinear
        self.to(device, non_blocking=True)

    def forward(self, obs):
        return torch.squeeze(self.v_net(torch.as_tensor(obs).to(device, non_blocking=True)), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(64,64),
        activation=nn.Tanh,
        pi_n_layers=2,
        v_n_layers=2,
        use_action_norm=False,
        ):
                 
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        # if isinstance(action_space, Box):
        #     self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        # elif isinstance(action_space, Discrete):
        #     self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        # self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        
        hidden1_dim = hidden_sizes[0]
        hidden2_dim = hidden_sizes[-1]
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        print(f'hidden1_dim: {hidden1_dim} ||| hidden2_dim: {hidden2_dim}')
        if use_action_norm:
            self.pi = MLPGaussianActor(
                obs_dim=state_dim, act_dim=action_dim, hidden_sizes=(hidden1_dim, hidden2_dim), n_layers=pi_n_layers, activation=nn.Mish,
                use_action_norm=use_action_norm
                ).to(device, non_blocking=True)
        else:
            self.pi = MLPGaussianActor(
                obs_dim=state_dim, act_dim=action_dim, hidden_sizes=(hidden1_dim, hidden2_dim), n_layers=pi_n_layers, activation=activation, # nn.Mish,
                use_action_norm=use_action_norm
                ).to(device, non_blocking=True)
        
        self.v = MLPCritic(
            obs_dim=state_dim, hidden_sizes=(hidden1_dim, hidden2_dim), n_layers=v_n_layers, activation=activation, # nn.Mish # activation
            ).to(device, non_blocking=True)
        print(f'self.pi: {self.pi} ||| self.v: {self.v}')
        # self.dummy_forward(state_dim=obs_dim, use_2d=False)
        self.to(device, non_blocking=True)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        # return a.numpy(), v.numpy(), logp_a.numpy()
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        # print(f' obs.shape: {obs.shape}')
        return self.step(obs)[0]

    def dummy_forward(self, state_dim, use_2d=False):
        if use_2d:
            with torch.no_grad():
                dummy = torch.randn(1000, *state_dim).to(device, non_blocking=True) # .unsqueeze(0)
                # _, a, _ = self.pi.sample(dummy, deterministic=False, with_logprob=True, act=) # [1]
                pi = self.actor_distribution(dummy) # , is_train=False)
                a = pi.sample()
                w = pi.probs
                print(f'w: {w} ||| w.shape: {w.shape} ||| a: {a} ||| a.shape: {a.shape} ||| pi: {pi}')
                # logp_a = self.pi._log_prob_from_distribution(pi, a) # Original.
                logp_a = self.pi._log_prob_from_distribution(pi, a) 
                print(f'||| logp_a: {logp_a} ||| logp_a.shape: {logp_a.shape}')
                v = self.value(dummy)
                print(f'a: {a} ||| v: {v}')
        else:
            with torch.no_grad():
                dummy = torch.randn(1000, state_dim).to(device, non_blocking=True) # .unsqueeze(0)
                # _, a, _ = self.pi.sample(dummy, deterministic=False, with_logprob=True, act=) # [1]
                a = self.pi(dummy) # , is_train=False)
                v = self.v(dummy)
                # print(f'a: {a} ||| v: {v}')



def pi_out_weight_init(m):
    """
    Custom weight initialization for Conv2D and Linear layers.

    This function applies orthogonal initialization to the weights of Linear layers and sets biases to zero.
    Orthogonal initialization can potentially lead to better convergence properties.

    :param m: A module (layer) of the neural network.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear) or isinstance(m, NoisyLazyLinear):
        # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh')) # Initialize weights using Xavier initialization
        # torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
        # Consistent orthogonal initialization
        nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('linear')) # 'tanh'))  # Apply orthogonal initialization to weights
        # print(f'nn.init Critic: {m}')
        if hasattr(m, 'bias') and m.bias is not None:
            # Initialize biases to zero
            m.bias.data.fill_(0)   # Set biases to zero
            # print(f'Set Bias Data Actor: {m.bias}')


def v_weight_init(m):
    """
    Custom weight initialization for Conv2D and Linear layers.

    This function applies orthogonal initialization to the weights of Linear layers and sets biases to zero.
    Orthogonal initialization can potentially lead to better convergence properties.

    :param m: A module (layer) of the neural network.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear): # or isinstance(m, NoisyLazyLinear):
        # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh')) # Initialize weights using Xavier initialization
        # torch.nn.init.kaiming_normal_(m.weight) # , nonlinearity='leaky_relu')

        # Consistent orthogonal initialization
        nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('linear'))  # Apply orthogonal initialization to weights
        # print(f'nn.init Critic: {m}')
        if hasattr(m, 'bias') and m.bias is not None:
            # Initialize biases to zero
            m.bias.data.fill_(0)   # Set biases to zero
            # print(f'Set Bias Data Actor: {m.bias}')


'''
PPO agent
'''
class PPOAgent(object):

    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    def __init__(
        self,
        env_fun,
        test_env_fun,
        hidden1_dim=256,
        hidden2_dim=256,
        gamma=0.99,
        steps_per_epoch=4000,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        target_kl=0.01,
        logger_kwargs=dict(),
        save_freq=1000,
        seed=0,
        pi_n_layers=2,
        v_n_layers=2,
        use_2d=False,
        use_action_norm=True,
        is_hyper_tune=False,
        ):

        self.is_hyper_tune = is_hyper_tune
        self.env = env_fun()
        self.test_env = test_env_fun()
        self.device = device
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.state_dim = state_dim
        self.delay = 1
        self.action_dim = action_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        # self.polyak = 1 - tau
        # self.scaler = GradScaler()
        # self.lambda_nll = 0.5
        # self.reward_scale = reward_scale
        self.actor_scaler = GradScaler()
        # self.beta = beta
        # self.alpha_init = alpha
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.use_2d = use_2d
        self.save_freq = save_freq
        self.gamma = gamma
        # self.tau = tau
        # self.batch_size = batch_size
        
        self.ac = MLPActorCritic(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_sizes=(hidden1_dim,hidden2_dim),
            activation=nn.Tanh, pi_n_layers=pi_n_layers, v_n_layers=v_n_layers,
            use_action_norm=use_action_norm,
            ).to(device, non_blocking=True)
        
        self.steps_per_epoch = steps_per_epoch # + 1
        # Set up experience buffer
        # self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs()) # Original.
        # self.local_steps_per_epoch = int(steps_per_epoch) # * num_procs())
        self.replay_buffer = PPOBuffer( # size, gamma=0.99, lam=0.95
            obs_dim=state_dim,
            act_dim=action_dim,
            size=self.steps_per_epoch, # Works.
            # size=self.local_steps_per_epoch,
            gamma=gamma,
            lam=lam,
            is_hyper_tune=self.is_hyper_tune,
            )
        
        # Set up optimizers for policy and value-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr) # , amsgrad=True) # RMSpropMax(self.ac.pi.parameters(), lr=lr) # Adam(self.ac.pi.parameters(), lr=lr, amsgrad=True)
        # self.pi_optimizer = RMSpropMax(self.policy_params, lr=lr)

        # Set up optimizers for policy and value function
        # pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        # self.ac.pi.optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr) # , betas=(0.3, 0.5))
        # vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.v_optimizer = Adam(self.ac.v.parameters(), vf_lr) # , betas=(0.3, 0.5))
        self.ac.dummy_forward(state_dim, use_2d=self.use_2d)

        self.ac.pi.apply(pi_out_weight_init)
        self.ac.v.apply(v_weight_init)

        # Set up model saving
        # logger.setup_pytorch_saver(ac)
        print(f' ||| self.v_optimizer.state_dict(): {self.v_optimizer.state_dict()}  |||')
        ################################

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi() # Original.

        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Seeding for reproducibility
        set_seed(seed=seed, env=self.env)
        set_seed(seed=seed, env=self.test_env)

        # Instantiate environment
        # env = env_fn()
        # obs_dim = env.observation_space.shape
        # act_dim = env.action_space.shape

        # Create actor-critic module
        # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

        if not self.is_hyper_tune:
            # Sync params across processes
            # sync_params(self.ac)
            sync_params(param=self.ac.parameters(), root=0) # 1) # Original.
        # self.logger.setup_pytorch_saver(self.ac)
        #  ppo_final.py
        # self.num_episodes_train

        # Count variables
        var_counts = tuple(count_vars(module) for module in [
            self.ac.pi, self.ac.v
            ]
                           )
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        logging.info('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        self.logger.setup_pytorch_saver(self.ac)
        
    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, debug=False):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # act = torch.as_tensor(act).to(device)
        # logp_old = torch.as_tensor(logp_old).to(device)
        # adv = torch.as_tensor(adv).to(device)
        
        # Get distribution and log probabilities
        pi, logp = self.ac.pi(obs, act)
        
        # Compute ratio for PPO clipping
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() # Original.

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        pi_info = dict(
            kl=approx_kl, ent=ent, cf=clipfrac
        )

        if debug:
            logging.info(
f'''
||| act.shape {act.shape}
||| logp.shape: {logp.shape}
||| logp_old.shape: {logp_old.shape}
||| ratio.shape: {ratio.shape}
||| ratio = torch.exp(logp - logp_old).mean(): {ratio.mean()}
||| (logp - logp_old).mean(): {(logp - logp_old).mean()}
||| adv.mean(): {adv.mean()}
||| clip_adv.mean() = (torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv).mean(): {clip_adv.mean()}
||| adv.shape: {adv.shape}
||| clip_adv.shape: {clip_adv.shape}
||| (ratio * adv).mean(): {(ratio * adv).mean()}
||| loss_pi = -(torch.min(ratio * adv, clip_adv)).mean(): {loss_pi}
'''
            )
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data, debug=False):
        obs, ret = data['obs'], data['ret']
        ret = torch.as_tensor(ret).to(device)
        v = self.ac.v(obs)
        if debug:
            loss1 = ((v - ret)**2).mean()
            loss2 = F.mse_loss(v, ret)
            logging.info(
f'''
||| obs.shape: {obs.shape}
||| ret.shape: {ret.shape}
||| v.shape: {v.shape}
||| loss1: {loss1}
||| loss2: {loss2}
'''
            )
            stop
        # return ((v - ret)**2).mean() # Original.
        # return F.mse_loss(self.ac.v(obs), ret)
        return F.mse_loss(v, ret) # , explained_var.mean().item()
        # v, _, _ = self.ac.v(obs)
        # return ((self.ac.v(obs) - ret)**2).mean()
        # loss += -logp.mean()
        # return loss

    def check_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                with torch.no_grad():
                    print_grad = param.grad.norm().item()
                if print_grad > 5. or print_grad < 1e-4:
                    logging.info(f"Layer {name} | Gradient Norm: {print_grad}")
                    # logging.info(f"Layer {name} | Gradient Norm: {param.grad.norm().item()}") # Original.
                    # logging.info(f"Layer {name} | Gradient Norm: {param.grad}")
            else:
                logging.info(f"Layer {name} | No Gradient")

    def remember(
        self, state, action, reward, new_state, done, value, log_probs, debug=False # , action_weights,
        ):
        # save and log
        # buf.store(o, a, r, v, logp)
        # self.logger.store(VVals=v)
        if debug:
            print(
# ||| action_weights:
# ||| {action_weights}
f'''
||| Remembering...
||| log_probs:
||| {log_probs}
||| log_probs.shape:
||| {log_probs.shape}
||| action:
||| {action}
||| value:
||| {value}
||| reward:
||| {reward}
'''
# ||| tanh action at env : {np.tanh(action)}
            )
        # self.replay_buffer.store(obs=state, act=action, rew=reward, next_obs=new_state, done=done, log_prob=log_probs) # Original and works.
        self.replay_buffer.store(
            obs=state,
            act=action,
            rew=reward,
            # val=value.item(), # Original.
            val=value,
            logp=log_probs, # Original.
            )
    def update(
        self, step, debug=False # , state, action, reward, next_state, done, step, value, log_probs
        ):
        if self.replay_buffer.ptr < self.replay_buffer.max_size: # self.batch_size: #
            print(
f'''
||| remembering...
||| self.replay_buffer.max_size:
||| {self.replay_buffer.max_size} 
||| self.replay_buffer.ptr
||| {self.replay_buffer.ptr}
||| action is:
||| {action}
||| reward is:
||| {reward}
||| returned!!!!!!!!
'''
            )
            return

        data = self.replay_buffer.get()

        pi_loss_old, pi_info_old = self.compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        value_loss_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            # loss_pi, kl = self.compute_loss_pi(data)
            if not self.is_hyper_tune:
                kl = mpi_avg(pi_info['kl']) # Original.
                # kl = mpi_avg_grads(pi_info['kl'])
            else:
                kl = pi_info['kl']
            # logging.info(f'1.5 * self.target_kl: {1.5 * self.target_kl} ||| kl: {kl}')
            # kl = kl
            if kl > 1.5 * self.target_kl:
                logging.info(f'1.5 * self.target_kl: {1.5 * self.target_kl} ||| kl: {kl}')
                logging.info(
                    'Early stopping at step %d due to reaching max kl.'%i
                    )
                self.logger.log(
                    'Early stopping at step %d due to reaching max kl.'%i
                    )
                break
            loss_pi.backward()
            if not self.is_hyper_tune:
                mpi_avg_grads(param_groups=self.pi_optimizer.param_groups) # Original.   # average grads across MPI processes
                # mpi_avg_grads(param_groups=self.ac.pi.parameters())
                # mpi_avg_grads(self.ac.pi)
            # if step % 3 == 0:
            self.check_gradients(self.ac.pi)
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            if not self.is_hyper_tune:
                # mpi_avg_grads(self.ac.v)    # average grads across MPI processes
                mpi_avg_grads(param_groups=self.v_optimizer.param_groups) # Original.
                # mpi_avg_grads(param_groups=self.ac.v.parameters())
            # if step % 3 == 0:
            self.check_gradients(self.ac.v)
            self.v_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf'] # cf for clipfrac
        self.logger.store(
            LossPi=pi_loss_old, LossV=value_loss_old,
            KL=kl, Entropy=ent, ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_loss_old),
            DeltaLossV=(loss_v.item() - value_loss_old),
            )

        y_true = data['ret']
        y_pred = data['val']
        var_y = np.var(y_true) # Original.
        # var_y = torch.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y # Original.
        # explained_var = np.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y
        if debug:
            logging.info(
f'''
||| explained_var: {explained_var}
'''
            )
        self.logger.store(
            ExplainedVariance = explained_var.item()
            )


def main():
# try:
    # import numpy as np
    # import torchrl # .modules.TanhNormal
    seed=42
    # torch.manual_seed(seed)
    # np.random.seed(seed)


    selected_assets = ['Close', 'ETH Close', 'SP500 Close']
    ### Parameters to tune ###
    hidden_dim = 256 # 512 # 128 # 64 # 1024 # 128 # 512 # 256 # 128 # 64 # 256 # 2048 # 4096 # 256 # 2048 # 64 # 2048 #1024 # 2048 #256 #1024 #4096
    hidden2_dim = 128 # 256 # 128 # 64 # 512 # 64 # 256 # 512 # 128 # 64 # 256 # 2048 # 64 # 2048 #1024 # 2048 #256
    gamma = 0.95 # 0.98 # 0.95 # 0.99
    # steps_per_epoch (int): Number of steps of interaction (state-action pairs)
    #         for the agent and the environment in each epoch.
    # steps_per_epoch = 4000
    # epochs=50
    clip_ratio = 0.2
    pi_lr = 3e-5 # 3e-6 # 3e-4 # 3e-3
    vf_lr = 1e-4 # 1e-5 # 1e-3 # 1e-5
    train_pi_iters = 120 # 1400 # 80
    train_v_iters = 120 # 1400 # 80
    lam = 0.97
    max_ep_len = 1000
    target_kl = 0.05 # 0.15 # 0.05 # 0.01
    # logger_kwargs=dict()
    save_freq = 25
    pi_n_layers = 4
    v_n_layers = 4
    ### End of Parameters to Tune ###
    
    initial_balance = 1000
    transaction_fee = 0.002
    num_episodes_train = 100 # 50 # 100 # 1000
    num_episodes_test = 3 # 10 # 20
    start_step_train = 5 # 7 # 4352 # 4567 # 3500 # 4200 # 4910 # 4567 # 4810 # 4910 # 4721 # 4717 # 4685 # 4567 # 4352 # 3987 # 3500 # 4000 
    start_step_test = 5 # 7 # 5 # 0 # 7
    use_2d = False
    train_ratio = 0.7 # 0.75 # 0.8
    use_action_norm = False # True
    ### Finished ###

    # Load and scale data
    # data = pd.read_csv('/home/jashha/merged_data_cleaned2.csv').fillna(0)
    # data = pd.read_parquet('/home/jashha/merged_data_cleaned2.parquet').fillna(0)
    # data = pd.read_csv('/home/jashha/merged_data_cleaned2.csv').replace([-np.inf, np.inf], np.nan).fillna(value=0) # .set_index('timestamp', drop=True) # .fillna(0).set_index('timestamp', drop=True)
    '''data = pd.read_parquet('/home/jashha/merged_data_cleaned3.parquet').fillna(value=0)
    col_to_drop = ['timestamp','Close', 'ETH Close', 'SP500 Close']
    price_data = pd.read_parquet('/home/jashha/merged_data.parquet').fillna(0).replace([-np.inf, np.inf], np.nan).fillna(value=0) # updated_features_df.parquet

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
    data = data[
            data.timestamp.between( 
                pd.to_datetime('20180101'), pd.to_datetime('20230616')
                )
            ]
    price_data = price_data[
            price_data.timestamp.between( 
                pd.to_datetime('20180101'), pd.to_datetime('20230616')
                )
            ]
    START_DATE = '20180101'
    END_DATE = '20230616'
    data.set_index('timestamp', drop=True, inplace=True)'''
    train_data_scaled, test_data_scaled, train_price_data, test_price_data, train_scaler, test_scaler = neutralization(data=None, start_step_train=start_step_train, train_ratio=train_ratio)
    # train_data_scaled, test_data_scaled, train_price_data, test_price_data, _, _ = neutralization(data=data, start_step_train=start_step_train, train_ratio=train_ratio)
    # print(f'||| train_data_scaled: {train_data_scaled} ||| train_data_scaled.shape: {train_data_scaled.shape} ||| test_data_scaled: {test_data_scaled} ||| test_data_scaled.shape: {test_data_scaled.shape}')

    '''price_cols = ['Close', 'ETH Close', 'SP500 Close', 'Adj Close', 'Open', 'High', 'Low', 'Volume', 'ETH Adj Close', 'ETH Open', 'ETH High', 'ETH Low', 'ETH Volume', 'SP500 Adj Close', 'SP500 Open', 'SP500 High', 'SP500 Low', 'SP500 Volume',]
    train_data_scaled, test_data_scaled, train_price_data, test_price_data, scaler = scale_dataframes_for_rl_env(data, price_data, scaler_type='minmax', train_ratio=train_ratio, start_date=START_DATE, end_date=END_DATE, price_cols=price_cols)
    train_scaler = scaler
    test_scaler = scaler
    print(f'||| train_data_scaled: {train_data_scaled} ||| train_data_scaled.shape: {train_data_scaled.shape} ||| test_data_scaled: {test_data_scaled} ||| test_data_scaled.shape: {test_data_scaled.shape}')'''

    reward_weights=dict(
        pf_sharpe=0.1,
        log_return=0.1,
        vol_monthly=0.05,
        pf_max_drawdown=0.05,
        alpha_monthly=0.1,
        m_squared_ratio=0.05,
        beta_adj_sharpe=0.5,
        pf_cvar_05=0.02,
        pf_sortino=0.1,
        pf_beta=0.1,
        blocked_actions_w=0, # 0.001,
        tx_fee_w=0.002,
        prospect_theory_loss_aversion_alpha=2.0,
        prospect_theory_loss_aversion_w=0.5,
        pf_parametric_var=0.02,
        base_ret=1.705,
        )

    env = CryptoTradingEnv( # CryptoTradingEnvv(
        train_data_scaled,
        # train_data,
        # res_train_data_scaled,
        selected_assets, initial_balance,
        transaction_fee, start_step_train, train_scaler,
        train=True, price_data=train_price_data,
        buffer_size=1000, env_name='env', use_2d=use_2d,
        if_discrete=False, use_action_norm=use_action_norm,
        reward_weights=reward_weights
        ) # (data_scaled, selected_assets, initial_balance, transaction_fee, start_step_train, train=True)

    test_env = CryptoTradingEnv( # CryptoTradingEnvv(
        # test_data,
        test_data_scaled,
        # test_data_scaled.drop(['Close', 'ETH Close', 'SP500 Close'], axis=1),
        # res_train_data_scaled,
        selected_assets, initial_balance,
        transaction_fee, start_step_test, test_scaler,
        train=False, price_data=test_price_data,
        buffer_size=1000, env_name='test_env', use_2d=use_2d,
        use_action_norm=use_action_norm,
        reward_weights=reward_weights,
        )

    # set_seed(seed=seed, env=env) # Original.
    # set_seed(seed=seed, env=test_env) # Original.
    # print(f'env: {env} ||| test_env: {test_env}')

    steps_per_epoch = env.max_ep_len # + 1 # * proc_id()
    epochs = num_episodes_train

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    # Prepare for interaction with environment
    start_time = time.time()
    
    logger_kwargs = setup_logger_kwargs(
        exp_name='ppo_continuous', seed=2024, data_dir='/home/jashha/', datestamp=False
        )
    
    if not use_2d:
        state_dim = len(env.get_state())
    else:
        state_dim = env.observation_space.sample().shape # [1]
    print(f'state_dim: {state_dim}')
    action_dim = len(selected_assets)

    ppo_agent = PPOAgent(
        env_fun=lambda: env, # gym.make(env),
        test_env_fun=lambda: test_env,
        # state_dim=state_dim,
        # action_dim=action_dim,
        hidden1_dim=hidden_dim,
        hidden2_dim=hidden2_dim,
        gamma=gamma,
        steps_per_epoch=env.max_ep_len, # steps_per_epoch,
        clip_ratio=clip_ratio,
        pi_lr=pi_lr,
        vf_lr=vf_lr,
        train_pi_iters=train_pi_iters,
        train_v_iters=train_v_iters,
        lam=lam,
        max_ep_len=max_ep_len,
        target_kl=target_kl,
        logger_kwargs=logger_kwargs,
        save_freq=save_freq,
        seed=0,
        pi_n_layers=pi_n_layers, v_n_layers=v_n_layers,
        use_2d=use_2d,
        use_action_norm=use_action_norm,
        is_hyper_tune=False,
        )

    print(f'||| ppo_agent is:  {ppo_agent} ||| ppo_agent.ac: {ppo_agent.ac}')

    # Define the directory to save models
    model_save_dir = '/home/jashha/checkpoints/' # '/home/jashha/' # '/home/jashha/YC/Models/' # '/home/jashha/'
    os.makedirs(model_save_dir, exist_ok=True)

    '''try:
        # ckpt = load_checkpoint("/home/jashha/ppo_continuous/ppo_continuous_s2024/pyt_save/checkpoint_10000.pt", device=device)
        ckpt = load_checkpoint("/home/jashha/checkpoints/checkpoint.pt", device=device)
        ppo_agent.ac.pi.load_state_dict(ckpt['actor_state'])
        ppo_agent.ac.v.load_state_dict(ckpt['value_state'])
        ppo_agent.pi_optimizer.load_state_dict(ckpt['actor_optimizer_state'])
        ppo_agent.v_optimizer.load_state_dict(ckpt['value_optimizer_state'])
        # if ckpt.get('scheduler_state') and scheduler:
        #     scheduler.load_state_dict(ckpt['scheduler_state'])
        # torch.set_rng_state(ckpt['rng_state'])
        # current_step = ckpt['step']
        
        print(f"[Info] Resumed from step {current_step}")
        print(f'ppo_agent.ac: {ppo_agent.ac}')
            # else:
            #     raise ValueError(f"Error loading models: {e}")
            #     # print(f"No saved model found for {model_name} at {model_path}.")

    except Exception as e:
        print(f"Error loading models: {e}")
        # raise ValueError(f"Error loading models: {e}")'''


    # If best model already exists, just load it
    model_dir  = "./results/training_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join( # /home/jashha/results/training_model/ppo_actor_critic_solid_base.pth
        model_dir, "ppo_actor_critic.pth" # "ppo_actor_critic_solid_base.pth" # "ppo_actor_critic.pth"
        )
    try:
        if os.path.exists(model_path):
            print(
                "Loading existing final model →", model_path
                )
            ppo_agent.ac.load_state_dict(
                torch.load(model_path)
                )
            
            print(f'||| epoch: {epoch} ||| ppo_agent.save_freq: {ppo_agent.save_freq} |||')
            # torch.save(
            #     ppo_agent.ac.state_dict(), model_path
            #     )
            # print(
            #     "Saved final model to", model_path
            #     )
        '''else:
            # print("Training final model with best hyper-parameters…")
            # train_ppo(
            #     final_agent, args.train_episodes, 0
            #     )
            torch.save(
                ppo_agent.ac.state_dict(), model_path
                )
            print(
                "Saved model to", model_path
                )'''
    except Exception as e:
        print(f"Error loading models: {e}")
        # raise ValueError(f"Error loading models: {e}")

    print(f'ppo_agent.ac: {ppo_agent.ac}')
    # stop
    # Paths for your buffer/scaler files
    REPLAY_BUFFER_PATH = "mlp_replay_buffer.pkl"
    SCALER_PATH = "mlp_scaler.pkl"
    RUNNING_SCALER_PATH = "mlp_train_running_scaler.pkl"

    # Example call:
    # running_scaler, _ = prefill_replay_buffer_and_scalers(
    running_scaler, scaler = prefill_replay_buffer_and_scalers(
    # running_scaler, scaler = prefill_replay_buffer_and_scalers(
        agent=ppo_agent,
        env=ppo_agent.env,
        test_env=test_env,
        replay_buffer=ppo_agent.replay_buffer,
        replay_buffer_path=REPLAY_BUFFER_PATH,
        scaler_path=SCALER_PATH,
        running_scaler_path=RUNNING_SCALER_PATH,
        replay_buffer_size=ppo_agent.env.max_ep_len, # replay_buffer_size,
        device=device,
        max_ep_len=ppo_agent.env.max_ep_len,
        logger=logging,
        is_ppo=True,
    )

    # env.set_running_scaler(running_scaler)
    test_env.set_running_scaler(running_scaler=running_scaler)
    ppo_agent.env.set_running_scaler(running_scaler=running_scaler)
    test_env.set_scaler(scaler=scaler)
    ppo_agent.env.set_scaler(scaler=scaler)

    del env
    del test_env

    train_ppo(ppo_agent, num_episodes_train, num_episodes_test)


if __name__ == "__main__":
    # mpi_fork(n=4) # , bind_to_core=True)
    main()
    # stop






        
