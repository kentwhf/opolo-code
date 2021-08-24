import gym
import numpy as np
import os
import random
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import TRPO, PPO2, SAC, TRPOGAIFO, OPOLO, HER
from stable_baselines.gail.dataset.dataset import ExpertDataset

from simulation_grounding.atp_envs import ATPEnv, GroundedEnv, MujocoNormalized, collect_trajectories
import matplotlib.pyplot as plt
import seaborn as sns


ALGO = HER
Grounding_ALGO = TRPOGAIFO
# set the environment here :
# REAL_ENV_NAME = 'Walker2dModified-v2'  # HopperFrictionModified-v2, Walker2dModified, InvertedPendulumModified
# SIM_ENV_NAME = 'Walker2d-v2'  # Hopper-v2, Walker2d, InvertedPendulum

REAL_ENV_NAME = 'FetchReach-v1'  
SIM_ENV_NAME = 'FetchReach-v1' 
PATH_PREFIX = 'opolo-baselines'


MUJOCO_NORMALIZE = True
ATP_NAME = '1'
INDICATOR = '_run1'
# RUN_NUM=5
# set this to the parent environment
TIME_STEPS = 2000000  # 2000000
SEED = 0
TOTAL_STEPS = 100000 # 100000

def plot_state_distributions(algo=ALGO):
    random.seed(SEED)
    np.random.seed(SEED)
    # test_policy = '../simulation_grounding/models/' + algo.__name__ + '_initial_policy_steps_' + SIM_ENV_NAME + '_' + str(
    #     TIME_STEPS) + '_.pkl'

    test_policy = PATH_PREFIX + "/run/test.zip"

    if algo.__name__ == 'PPO2':
        algo = PPO2
    elif algo.__name__ == 'TRPO':
        algo = TRPO
    elif algo.__name__ == 'SAC':
        algo = SAC

    sim_env = gym.make(SIM_ENV_NAME)
    real_env = gym.make(REAL_ENV_NAME)

    # if MUJOCO_NORMALIZE:
    #     sim_env = MujocoNormalized(sim_env)
    #     real_env = MujocoNormalized(real_env)
    # atp_policy = '../run/tmp/logs/trpo-gaifo/trpogaifo/'+ SIM_ENV_NAME + '/rank1/action_transformer_policy1.pkl'
    # atp_policy = '../run/tmp/logs/td3-opolo-idm-decay-reg/opolo/' + SIM_ENV_NAME + '/rank1/action_transformer_policy1.pkl'

    atp_policy = PATH_PREFIX + "/run/test/logs/trpo-gaifo/trpogaifo/FetchReach-v1/rank0/action_transformer_policy1.pkl"

    # atp_policy = '../run/first_trial0317/logs/td3-opolo-idm-decay-reg/opolo/'+ SIM_ENV_NAME + '/rank'+str(ATP_NAME)+'/action_transformer_policy1.pkl'
    # atp_policy = '../simulation_grounding/garat_baselines/' + ATP_NAME + '/grounding_step_0/action_transformer_policy1_49.pkl'

    # atp_policy = '/home/hyun-rok/Documents/Exp_Multiruns/WIP-safe-transfer-RL/data/models/garat/' + ATP_NAME + '/grounding_step_0/action_transformer_policy1_49.pkl'

    if Grounding_ALGO.__name__ == 'TRPO':
        atp_environment = TRPO.load(load_path=atp_policy, seed=SEED+100)
        scale_atp = False
    elif Grounding_ALGO.__name__ == 'PPO2':
        config = {'expert_data_path': None}
        atp_environment = PPO2.load(load_path=atp_policy, seed=SEED+100, config=config)
        scale_atp = False
    elif Grounding_ALGO.__name__ == 'TRPOGAIFO':
        config = {'expert_data_path': None}
        atp_environment = TRPOGAIFO.load(load_path=atp_policy, seed=SEED+100, config=config)
        scale_atp = False
    elif Grounding_ALGO.__name__ == 'OPOLO':
        config = {'expert_data_path': None, 'shaping_mode': 'td3-opolo-idm-decay-reg'}
        atp_environment = OPOLO.load(load_path=atp_policy, seed=SEED+100, config=config)
        scale_atp = True

    #########################################################################
    use_deterministic = False
    grounded_env = GroundedEnv(env=sim_env,
                               action_tf_policy=atp_environment,
                               debug_mode=True,
                               data_collection_mode=False,
                               use_deterministic=use_deterministic,
                               atp_policy_noise=0.0,
                               scale_atp=scale_atp,
                               )

    # 1. Collect real trajectory and get transition errors
    real_env.seed(SEED)
    real_Ts, real_rews, transition_errors = collect_trajectories(env=real_env,
                                                                 policy=algo.load(load_path=test_policy, seed=SEED+1000, env=real_env),
                                                                 limit_trans_count=TOTAL_STEPS,
                                                                 deterministic=False,
                                                                 transition_errors=True,
                                                                 grounded_env=grounded_env)
    real_Ts = np.array(real_Ts)

    print(ATP_NAME)

    sim_env.seed(SEED)
    sim_Ts, sim_rews, _ = collect_trajectories(env=sim_env,
                                               policy=algo.load(load_path=test_policy, seed=SEED+1000, env=sim_env),
                                               limit_trans_count=TOTAL_STEPS,
                                               deterministic=False,
                                               )
    sim_Ts = np.array(sim_Ts)

    grounded_env.seed(SEED)
    grounded_env.reset_saved_actions()
    grounded_Ts, grnd_rews, _ = collect_trajectories(env=grounded_env,
                                                     policy=algo.load(load_path=test_policy, seed=SEED+1000, env=grounded_env),
                                                     limit_trans_count=TOTAL_STEPS,
                                                     deterministic=False,
                                                     )
    mean_delta, max_delta = grounded_env.plot_action_transformation(expt_path=SIM_ENV_NAME + '_' + Grounding_ALGO.__name__ + '_' + ALGO.__name__ + INDICATOR + '_action_transformer.png',
                                            max_points=10000)
    grounded_Ts = np.array(grounded_Ts)

    # Write results files
    with open('results_' + SIM_ENV_NAME + '_' + Grounding_ALGO.__name__ + '_' + INDICATOR + '.txt', 'w') as f:
        f.write("Episodic returns (Src to Real, Sim, Grnd) \n")
        np.savetxt(f, (real_rews, sim_rews, grnd_rews))
        f.write("Transition errors \n")
        np.savetxt(f, [transition_errors])
        f.write("Mean, max delta \n")
        np.savetxt(f, (mean_delta, max_delta))
        f.close()

    for i in range(np.shape(real_Ts)[1]):
        # for i in range(1):
        p1 = sns.kdeplot(real_Ts[:, i], shade=True, color="r", label="target")
        p1 = sns.kdeplot(sim_Ts[:, i], shade=True, color="b", label="source")
        p1 = sns.kdeplot(grounded_Ts[:, i], shade=True, color="g", label="grounded")
        plt.legend()
        plt.savefig(
            SIM_ENV_NAME + '_' + Grounding_ALGO.__name__ + '_' + ALGO.__name__ + '_dimension' + str(i + 1) + INDICATOR + '.png')
        plt.close()
        # plt.show()
    os._exit(0)

if __name__ == '__main__':
    plot_state_distributions()

    # # Plot expert trajectories
    # save_path = '../simulation_grounding/real_traj'
    # data_save_dir = os.path.join(save_path, "{}_episodes_{}".format(REAL_ENV_NAME, 4))
    # expert_data_path = data_save_dir + ".npz"
    # expert_dataset = ExpertDataset(expert_path=expert_data_path, ob_flatten=False)
    # for i in range(np.shape(expert_dataset.observations)[1] - 1): # 1 = action_dimension
    #     p1 = sns.kdeplot(expert_dataset.observations[:, i], shade=True, color="y", label="expert")
    #     plt.show()

    os._exit(0)