import argparse
import os
from collections import OrderedDict
from pprint import pprint
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')
import tensorflow as tf
# For pybullet envs
warnings.filterwarnings("ignore")
import gym
import mujoco_py
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
import yaml

from stable_baselines import TRPO, OPOLO, TRPOGAIFO, PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

from simulation_grounding.atp_envs import GroundedEnv, MujocoNormalized

best_mean_reward, n_steps = -np.inf, 0

def eval_real_callback(log_dir, eval_real_env, eval_grnd_env, n_eval_episodes = 5):
    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps
        # Print stats every 20 calls
        if (n_steps + 1) % 1 == 0:
            # Evaluate policy training performance
            episode_rewards, episode_lengths = evaluate_policy(_locals['self'], eval_real_env,
                                                               n_eval_episodes=n_eval_episodes,
                                                               render=False,
                                                               deterministic=False,
                                                               return_episode_rewards=False)
            print("Last mean reward per episode at target: {:.2f}".format(episode_rewards))

            episode_rewards_grnd, episode_lengths_grnd = evaluate_policy(_locals['self'], eval_grnd_env,
                                                               n_eval_episodes=n_eval_episodes,
                                                               render=False,
                                                               deterministic=False,
                                                               return_episode_rewards=False)
            print("Last mean reward per episode at grounded environment: {:.2f}".format(episode_rewards_grnd))

            with open(os.path.join(log_dir, 'eval_at_target.txt'), 'a') as f:
                f.write("{}, {}, {}\n".format(n_steps, episode_rewards, episode_lengths/n_eval_episodes))
                f.close()
            with open(os.path.join(log_dir, 'eval_at_grnd.txt'), 'a') as f:
                f.write("{}, {}, {}\n".format(n_steps, episode_rewards_grnd, episode_lengths_grnd/n_eval_episodes))
                f.close()
        n_steps += 1
        return True
    return callback

def evaluate_policy_on_env(env,
                           model,
                           render=False,
                           iters=50,
                           deterministic=False
                           ):
    return_list = []
    for i in range(iters):
        return_val = 0
        done = False
        obs = env.reset()
        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = env.step(action)
            return_val+=rewards
            if render:
                env.render()

        if not i%15: print('Iteration ', i, ' done.')
        return_list.append(return_val)
    print('***** STATS FOR THIS RUN *****')
    print('MEAN : ', np.mean(return_list))
    print('STD : ', np.std(return_list))
    return np.mean(return_list), np.std(return_list)/np.sqrt(len(return_list)), 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', default='Hopper-v2', help="Name of the simulator environment (Unmodified)")
    # parser.add_argument('--real_env', default='HopperFrictionModified-v2', help="Name of the Real World environment (Modified)")
    # parser.add_argument('--rollout_policy_path', default="../simulation_grounding/models/TRPO_initial_policy_steps_Hopper-v2_2000000_.pkl", help="relative path of initial policy trained in sim")

    # parser.add_argument('--env', default='Walker2d-v2', help="Name of the simulator environment (Unmodified)")
    # parser.add_argument('--real_env', default='Walker2dModified-v2', help="Name of the Real World environment (Modified)")
    # parser.add_argument('--rollout_policy_path', default="../simulation_grounding/models/TRPO_initial_policy_steps_Walker2d-v2_2000000_.pkl", help="relative path of initial policy trained in sim")

    parser.add_argument('--env', default='InvertedPendulum-v2', help="Name of the simulator environment (Unmodified)")
    parser.add_argument('--real_env', default='InvertedPendulumModified-v2', help="Name of the Real World environment (Modified)")
    parser.add_argument('--rollout_policy_path', default="../simulation_grounding/models/TRPO_initial_policy_steps_InvertedPendulum-v2_1000000_.pkl", help="relative path of initial policy trained in sim")

    parser.add_argument('--ground_algo', help='Grounding Algorithm', default='TRPOGAIFO', type=str) # TRPO, PPO2, TRPOGAIFO, OPOLO
    parser.add_argument('--atp_policy_path', default="../run/test/logs/trpo-gaifo/trpogaifo/InvertedPendulum-v2/rank1/action_transformer_policy1.pkl", type=str, help='relative path of action transformer policy')
    # parser.add_argument('--atp_policy_path', default="../run/test/logs/td3-opolo-idm-decay-reg/opolo/InvertedPendulum-v2/rank1/action_transformer_policy1.pkl", type=str, help='relative path of action transformer policy')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)

    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=1000, type=int)
    parser.add_argument('--log-dir', help='Log directory', type=str, default='../simulation_grounding/target_policies/') # required=True,
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,type=int)
    args = parser.parse_args()

    # extend log directory with experiment details
    new_log_dir = os.path.join(args.log_dir, args.env, args.ground_algo, 'rank{}'.format(args.seed))
    args.log_dir = new_log_dir

    ################################################
    set_global_seeds(args.seed)
    tensorboard_log = os.path.join(args.log_dir, 'tb')

    print("=" * 10, args.env, "=" * 10)
    os.makedirs(args.log_dir, exist_ok=True)

    #####################################
    # Load model
    #####################################
    print('LOADING -PRETRAINED- INITIAL POLICY')
    with open('../hyperparams/trpo.yml') as file:
        policy_params = yaml.load(file, Loader=yaml.FullLoader)

    print('Using TRPO as the Target Policy Algo')
    policy_params = policy_params[args.env]

    # Create grounded environment
    if args.ground_algo == 'TRPO':
        atp_environment = TRPO.load(load_path=args.atp_policy_path, seed=args.seed+100)
        scale_atp = False
    elif args.ground_algo == 'PPO2':
        config = {'expert_data_path': None}
        atp_environment = PPO2.load(load_path=args.atp_policy_path, seed=args.seed+100, config=config)
        scale_atp = False
    elif args.ground_algo == 'TRPOGAIFO':
        config = {'expert_data_path': None}
        atp_environment = TRPOGAIFO.load(load_path=args.atp_policy_path, seed=args.seed+100, config=config)
        scale_atp = False
    elif args.ground_algo == 'OPOLO':
        config = {'expert_data_path': None, 'shaping_mode': 'td3-opolo-idm-decay-reg'}
        atp_environment = OPOLO.load(load_path=args.atp_policy_path, seed=args.seed+100, config=config)
        scale_atp = True

    sim_env = gym.make(args.env)
    if 'env_wrapper' in policy_params.keys():
        if 'MujocoNormalized' in policy_params['env_wrapper']:
            sim_env = MujocoNormalized(sim_env)

    use_deterministic = False
    grnd_env = GroundedEnv(env=sim_env,
                           action_tf_policy=atp_environment,
                           debug_mode=False,
                           data_collection_mode=False,
                           use_deterministic=use_deterministic,
                           atp_policy_noise=0.0,
                           scale_atp=scale_atp,
                           )
    grnd_env.seed(args.seed)

    ### SET CALLBACK
    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    sim_eval_env = gym.make(args.env)
    if 'env_wrapper' in policy_params.keys():
        if 'MujocoNormalized' in policy_params['env_wrapper']:
            sim_eval_env = MujocoNormalized(sim_eval_env)

    use_deterministic = False
    grnd_eval_env = GroundedEnv(env=sim_eval_env,
                           action_tf_policy=atp_environment,
                           debug_mode=False,
                           data_collection_mode=False,
                           use_deterministic=use_deterministic,
                           atp_policy_noise=0.0,
                           scale_atp=scale_atp,
                           )
    grnd_eval_env.seed(args.seed)

    real_callback_env = gym.make(args.real_env)
    real_callback_env.seed(args.seed)
    if 'env_wrapper' in policy_params.keys():
        if 'MujocoNormalized' in policy_params['env_wrapper']:
            real_callback_env = MujocoNormalized(real_callback_env)
    cb_func = eval_real_callback(log_dir=args.log_dir, eval_real_env = real_callback_env, eval_grnd_env = grnd_eval_env)

    ##### Load source policy
    model = TRPO.load(
        args.rollout_policy_path,
        seed=args.seed,
        env=DummyVecEnv([lambda:grnd_env]),
        verbose=args.verbose,
        # disabled tensorboard temporarily
        # tensorboard_log=None,
        tensorboard_log= tensorboard_log,
        timesteps_per_batch=policy_params['timesteps_per_batch'],
        lam=policy_params['lam'],
        max_kl=policy_params['max_kl'],
        gamma=policy_params['gamma'],
        vf_iters=policy_params['vf_iters'],
        vf_stepsize=policy_params['vf_stepsize'],
        entcoeff=policy_params['entcoeff'],
        cg_damping=policy_params['cg_damping'],
        cg_iters=policy_params['cg_iters']
    )

    ##### Learn policy
    n_timesteps = int(policy_params['n_timesteps'])
    model.learn(
        total_timesteps=n_timesteps,
        callback=cb_func,
        reset_num_timesteps=True,
        **kwargs)
    model.save(os.path.join(args.log_dir, 'target_policy.pkl'))

    with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
        yaml.dump(policy_params, f)

    ##### Evaluate policy in target environment
    real_env = gym.make(args.real_env)
    if 'env_wrapper' in policy_params.keys():
        if 'MujocoNormalized' in policy_params['env_wrapper']:
            real_env = MujocoNormalized(real_env)

    real_env.seed(args.seed)
    val = evaluate_policy_on_env(real_env,
                                 model,
                                 render=False,
                                 iters=50,
                                 deterministic=True
                                 )
    with open(args.log_dir + "/output.txt", "a") as txt_file:
        print(val, file=txt_file)

    real_env.seed(args.seed)
    val = evaluate_policy_on_env(real_env,
                                 model,
                                 render=False,
                                 iters=50,
                                 deterministic=False
                                 )
    with open(args.log_dir + "/stochastic_output.txt", "a") as txt_file:
        print(val, file=txt_file)