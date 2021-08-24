from cloudpickle.cloudpickle import instance
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mujoco_py
from stable_baselines import TRPO, PPO2, SAC

from stable_baselines import HER
from gym import spaces

from simulation_grounding import *
from stable_baselines.common.math_util import unscale_action, scale_action

# Ratio between the max change in action by the action transformer and the
# action range of the actual simulator (smaller values are more restrictive
# but lead to faster learning)
ACTION_TF_RATIO = 2.0


def get_parent_env(env_name):
    parent_env_dict = {
        'Hopper-v2': 'Hopper-v2',
        'HopperModified-v2': 'Hopper-v2',
        'HopperArmatureModified-v2': 'Hopper-v2',
        'HopperFrictionModified-v2': 'Hopper-v2',
        'DartHopper-v1': 'Hopper-v2',
        'InvertedPendulumModified-v2': 'InvertedPendulum-v2',
        'InvertedPendulum-v2': 'InvertedPendulum-v2',
        'Walker2dModified-v2': 'Walker2d-v2',
        'Walker2dFrictionModified-v2': 'Walker2d-v2',
        'Walker2d-v2': 'Walker2d-v2',
        'DartWalker2d-v1': 'Walker2d-v2',
        'MinitaurRealBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurRealOnRackBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurRealBulletEnvRender-v0': 'MinitaurBulletEnv-v0',
        'MinitaurInaccurateMotorBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurInaccurateMotorOnRackBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'MinitaurInaccurateMotorBulletEnvRender-v0': 'MinitaurBulletEnv-v0',
        # 'Ant-v2': 'Ant-v2',
        # 'AntLowGravity-v2': 'Ant-v2',
        'HalfCheetah-v2': 'HalfCheetah-v2',
        'DartHalfCheetah-v1': 'HalfCheetah-v2',
        'AntPyBulletEnv-v0': 'AntPyBulletEnv-v0',
        'AntModifiedBulletEnv-v0': 'AntPyBulletEnv-v0',
        'MinitaurBulletEnv-v0': 'MinitaurBulletEnv-v0',
        'FetchReach-v1': 'FetchReach-v1'
    }

    if env_name not in parent_env_dict.keys():
        raise ValueError(
            'The environment has not been added to the mapping yet. Please check scripts/utils/get_parent_env.py')

    return parent_env_dict[env_name]


def set_rollout_policy(rollout_policy_path, seed=None):
    # if 'TRPO' in rollout_policy_path:
    #     algo = TRPO
    # elif 'PPO2' in rollout_policy_path:
    #     algo = PPO2
    # elif 'SAC' in rollout_policy_path:
    #     algo = SAC
    # else:
    #     raise NotImplementedError("Algorithm for rollout policy is not supported yet")
    # return TRPO.load(rollout_policy_path, seed=seed)

    from gym.envs.robotics.fetch.reach import FetchReachEnv
    environment = FetchReachEnv()
    return HER.load(rollout_policy_path, env=environment, seed=seed)


def collect_trajectories(env, policy, limit_trans_count, deterministic, transition_errors=False, grounded_env=None):
    Ts = []  # Initialize list of trajectories
    action_lim = abs(env.action_space.high)
    l2_error_list = []
    episodic_rewards = []

    print('COLLECTING TRAJECTORIES ... ')
    transition_count = 0
    while True:
        done = False
        obs = env.reset()

        if isinstance(policy, HER):
            obs = np.concatenate([obs["achieved_goal"],
                                  obs["observation"],
                                  obs["desired_goal"]])

        if transition_errors: grounded_env.reset()
        Ts.append(obs)
        epi_rew = 0
        while not done:
            if transition_errors: grounded_env.reset_state_to_real(env.unwrapped.sim.get_state(), obs)

            action, _ = policy.predict(obs, deterministic=deterministic)

            # clip action within range
            action = np.clip(action, -action_lim, action_lim)
            obs, rew, done, _ = env.step(action)
            epi_rew += rew

            if isinstance(policy, HER):
                obs = np.concatenate([obs["achieved_goal"],
                                        obs["observation"],
                                        obs["desired_goal"]])
            
            Ts.append(obs)

            if transition_errors:
                grounded_obs, _, done_grounded, _ = grounded_env.step(action)

                if isinstance(policy, HER):
                    grounded_obs = np.concatenate([grounded_obs["achieved_goal"],
                                                    grounded_obs["observation"],
                                                    grounded_obs["desired_goal"]])
                
                l2_error_list.append(np.linalg.norm(grounded_obs - obs))

            transition_count += 1
        episodic_rewards.append(epi_rew)
        if transition_count > limit_trans_count:
            print('~~ STOPPING COLLECTING EXPERIENCE ~~')
            print('Average episodic rewards: ', np.mean(episodic_rewards))
            break
    if transition_errors:
        print('Transition error', np.mean(l2_error_list))
        return Ts, np.mean(episodic_rewards), np.mean(l2_error_list)
    else:
        return Ts, np.mean(episodic_rewards), 0.0

class ATPEnv(gym.Wrapper):
    """
    Defines the Action Transformer Policy's environment
    """
    def __init__(self,
                 env,
                 rollout_policy_path,
                 atr=ACTION_TF_RATIO,
                 train_noise=0.0,
                 frames=1,
                 seed=None,
                 is_OPOLO=False,
                 ):
        super(ATPEnv, self).__init__(env)
        self.rollout_policy = set_rollout_policy(rollout_policy_path, seed = seed)

        # Dimension concatenation and reduction
        # low = np.concatenate([self.env.observation_space["achieved_goal"].low,
        #                       self.env.observation_space["observation"].low,
        #                       self.env.observation_space["desired_goal"].low,
        #                       self.env.action_space.low])
        # high = np.concatenate([self.env.observation_space["achieved_goal"].high,
        #                        self.env.observation_space["observation"].high,
        #                        self.env.observation_space["desired_goal"].high,
        #                        self.env.action_space.high])

        low = np.concatenate([self.env.observation_space["achieved_goal"].low,
                            #   self.env.observation_space["desired_goal"].low,
                              self.env.action_space.low])
        high = np.concatenate([self.env.observation_space["achieved_goal"].high,
                            #    self.env.observation_space["desired_goal"].high,
                               self.env.action_space.high])

        self.obs_size = low.shape[0]
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        low = self.env.action_space.low
        high = self.env.action_space.high
        self.env_max_act = (self.env.action_space.high - self.env.action_space.low) / 2

        self.is_OPOLO = is_OPOLO
        max_act = (self.env.action_space.high - self.env.action_space.low) / 2 * atr
        if is_OPOLO:
            self.action_space = spaces.Box(-max_act / max_act, max_act / max_act, dtype=np.float32)
        else:
            self.action_space = spaces.Box(-max_act, max_act, dtype=np.float32)

        self.train_noise = train_noise
        self.frames = frames

        # These are set when reset() is called
        self.latest_obs = None
        self.latest_act = None
        self.prev_frames = None

    def reset(self, **kwargs):
        """Reset function for the wrapped environment"""
        self.latest_obs = self.env.reset(**kwargs)

        self.latest_act, _ = self.rollout_policy.predict(self.latest_obs, deterministic=False)

        # create empty list and pad with zeros
        self.prev_frames = []
        self.prev_actions = []
        for _ in range(self.frames - 1):
            self.prev_frames.extend(np.zeros_like(np.hstack((self.latest_obs, self.latest_act))))
        self.prev_frames.extend(np.hstack((self.latest_obs, self.latest_act)))

        # Return the observation for THIS environment
        return np.append(self.latest_obs, self.latest_act)

    def step(self, action):
        """
        Step function for the wrapped environment
        """
        # input action is the delta transformed action for this Environment
        if self.is_OPOLO:
            # Unscale input action [(-1,1) to original action space]
            max_act = (self.env.action_space.high - self.env.action_space.low) / 2 * ACTION_TF_RATIO
            action = unscale_action(spaces.Box(-max_act, max_act, dtype=np.float32), action)

        # print(action)
        transformed_action = action + self.latest_act
        transformed_action = np.clip(transformed_action, -self.env_max_act, self.env_max_act)

        sim_next_state, sim_rew, sim_done, info = self.env.step(transformed_action)

        # get target policy action
        target_policy_action, _ = self.rollout_policy.predict(sim_next_state, deterministic=False)

        ###### experimenting with adding noise while training ATPEnv ######
        # target_policy_action = target_policy_action + np.random.normal(0, self.train_noise**0.5, target_policy_action.shape[0])

        concat_sa = np.append(sim_next_state, target_policy_action)

        self.latest_obs = sim_next_state
        self.latest_act = target_policy_action

        self.prev_frames = self.prev_frames[self.obs_size:]
        self.prev_frames.extend(np.hstack((self.latest_obs, self.latest_act)))

        return concat_sa, sim_rew, sim_done, info

    def close(self):
        self.env.close()

class GroundedEnv(gym.ActionWrapper):
    """
    Defines the grounded environment, from the perspective of the target policy
    """
    # pylint: disable=abstract-method
    def __init__(self,
                 env,
                 action_tf_policy,
                 # action_tf_env,
                 alpha=1.0,
                 debug_mode=True,
                 normalizer=None,
                 data_collection_mode=False,
                 use_deterministic=True,
                 atp_policy_noise=0.0,
                 scale_atp=False,
                 ):
        super(GroundedEnv, self).__init__(env)
        self.debug_mode = debug_mode
        self.action_tf_policy = action_tf_policy
        self.alpha = alpha
        self.normalizer = normalizer
        if self.debug_mode:
            self.transformed_action_list = []
            self.raw_actions_list = []
        if data_collection_mode:
            self.data_collection_mode = True
            self.Ts = []
        else:
            self.data_collection_mode = False
        # These are set when reset() is called
        self.latest_obs = None
        # self.prev_frames = None
        self.time_step_counter = 0
        self.high = env.action_space.high
        self.low = env.action_space.low

        max_act = (self.high - self.low) / 2 * ACTION_TF_RATIO
        self.transformed_action_space = spaces.Box(-max_act, max_act, dtype=np.float32)
        self.use_deterministic = use_deterministic
        self.atp_policy_noise = atp_policy_noise
        self.scale_atp = scale_atp

    def reset(self, **kwargs):
        if self.normalizer is not None:
            # self.latest_obs = self.normalizer.normalize_obs(self.latest_obs)
            self.latest_obs = self.normalizer.reset(**kwargs)
            self.latest_obs = self.latest_obs[0]
        else:
            self.latest_obs = self.env.reset(**kwargs)

        if self.data_collection_mode:
            self.T = []

        # self.prev_frames = [self.latest_obs for _ in range(NUM_FRAMES_INPUT)]
        self.time_step_counter = 0
        return self.latest_obs

    def reset_state_to_real(self, real_sim_state, real_obs):
        new_state = mujoco_py.MjSimState(real_sim_state.time, real_sim_state.qpos, real_sim_state.qvel,
                                         real_sim_state.act, real_sim_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

        self.latest_obs = real_obs

        # if 'InvertedPendulum' in self.env.unwrapped.spec.id:
        #     self.latest_obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        # elif 'Walker' in self.env.unwrapped.spec.id:
        #     qpos = self.sim.data.qpos
        #     qvel = self.sim.data.qvel
        #     self.latest_obs = np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        # elif 'Hopper' in self.env.unwrapped.spec.id:
        #     self.latest_obs = np.concatenate([self.sim.data.qpos.flat[1:],
        #                                       np.clip(self.sim.data.qvel.flat, -10, 10)])
        # elif 'HalfCheetah' in self.env.unwrapped.spec.id:
        #     self.latest_obs = np.concatenate([self.sim.data.qpos.flat[1:],
        #                                       self.sim.data.qvel.flat,])
        # else:
        #     assert "Unsupported gym environment"

    def step(self, action):
        self.time_step_counter += 1

        if self.data_collection_mode: self.T.append((self.latest_obs, action))

        # TODO: add more frames here ?
        concat_sa = np.append(self.latest_obs, action)
        # change made : lets assume the output of the ATP is \delta_a_t
        # concat_sa = self.atp_env.normalize_obs(concat_sa)
        if isinstance(self.action_tf_policy, list):
            delta_transformed_action, _ = self.action_tf_policy[0].predict(concat_sa,
                                                                           deterministic=self.use_deterministic)
            for i in range(1, len(self.action_tf_policy)):
                delta_transformed_action_i, _ = self.action_tf_policy[i].predict(concat_sa,
                                                                                 deterministic=self.use_deterministic)
                delta_transformed_action += delta_transformed_action_i
            delta_transformed_action = delta_transformed_action / len(self.action_tf_policy)
        else:
            if concat_sa.shape[0] != 20:
                from itertools import chain
                act = concat_sa[1:].reshape(1, -1)
                obs = np.asarray([list(chain.from_iterable(concat_sa[0].values()))])
                concat_sa = np.concatenate((obs,act), axis=1).astype('float32').reshape(-1)

                # print("="*10)
                # print(concat_sa)
                # print(concat_sa.shape)
                # print("="*10)

            delta_transformed_action, _ = self.action_tf_policy.predict(concat_sa, deterministic=self.use_deterministic)

        if self.scale_atp:
            delta_transformed_action = unscale_action(self.transformed_action_space, delta_transformed_action)
        # #NEW : experimenting with adding noise here
        # delta_transformed_action += np.random.normal(0, self.atp_policy_noise**0.5, delta_transformed_action.shape[0])

        # print('delta : ',delta_transformed_action)

        transformed_action = action + self.alpha * delta_transformed_action
        # transformed_action = action + delta_transformed_action

        transformed_action = np.clip(transformed_action, self.low, self.high)

        # transformed_action = delta_transformed_action
        if self.normalizer is not None:
            self.latest_obs, rew, done, info = self.normalizer.step(transformed_action)
            self.latest_obs = self.latest_obs[0]
            rew, done, info = rew[0], done[0], info[0]
        else:
            self.latest_obs, rew, done, info = self.env.step(transformed_action)

        # if self.normalizer is not None:
        #     self.latest_obs = self.normalizer.normalize_obs(self.latest_obs)
        # self.prev_frames = self.prev_frames[1:]+[self.latest_obs]

        if self.debug_mode and self.time_step_counter <= 1e4:
            self.transformed_action_list.append(transformed_action)
            self.raw_actions_list.append(action)

        # change the reward to be a function of the input action and
        # not the transformed action
        if 'Hopper' in self.env.unwrapped.spec.id:
            rew = rew - 1e-3 * np.square(action).sum() + 1e-3 * np.square(transformed_action).sum()
        elif 'HalfCheetah' in self.env.unwrapped.spec.id:
            rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()
        elif 'Swimmer' in self.env.unwrapped.spec.id:
            rew = rew - 0.0001 * np.square(action).sum() + 0.0001 * np.square(transformed_action).sum()
        elif 'Reacher' in self.env.unwrapped.spec.id:
            rew = rew - np.square(action).sum() + np.square(transformed_action).sum()
        elif 'Ant' in self.env.unwrapped.spec.id:
            rew = rew - 0.5 * np.square(action).sum() + 0.5 * np.square(transformed_action).sum()
        elif 'Humanoid' in self.env.unwrapped.spec.id:
            rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()
        elif 'Pusher' in self.env.unwrapped.spec.id:
            rew = rew - np.square(action).sum() + np.square(transformed_action).sum()
        elif 'Walker2d' in self.env.unwrapped.spec.id:
            rew = rew - 1e-3 * np.square(action).sum() + 1e-3 * np.square(transformed_action).sum()
        elif 'HumanoidStandup' in self.env.unwrapped.spec.id:
            rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()

        if done and self.data_collection_mode:
            self.T.append((self.latest_obs, None))
            self.Ts.extend(self.T)
            self.T = []

        return self.latest_obs, rew, done, info

    def get_trajs(self):
        return self.Ts

    def reset_saved_actions(self):
        self.transformed_action_list = []
        self.raw_actions_list = []

    def plot_action_transformation(
            self,
            expt_path,
            show_plot=True,
            max_points=3000):
        """Graphs transformed actions vs input actions"""
        num_action_space = self.env.action_space.shape[0]
        action_low = self.env.action_space.low[0]
        action_high = self.env.action_space.high[0]

        self.raw_actions_list = np.asarray(self.raw_actions_list)
        self.transformed_action_list = np.asarray(self.transformed_action_list)

        mean_delta = np.mean(np.abs(self.raw_actions_list - self.transformed_action_list))
        max_delta = np.max(np.abs(self.raw_actions_list - self.transformed_action_list))
        print("Mean delta transformed_action: ", mean_delta)
        print("Max:", max_delta)
        # Reduce sample size
        index = np.random.choice(np.shape(self.raw_actions_list)[0], max_points, replace=False)
        self.raw_actions_list = self.raw_actions_list[index]
        self.transformed_action_list = self.transformed_action_list[index]

        colors = ['go', 'bo', 'ro', 'mo', 'yo', 'ko', 'go']

        if num_action_space > len(colors):
            print("Unsupported Action space shape.")
            return

        # plotting the data points starts here
        fig = plt.figure(figsize=(int(10 * num_action_space), 10))
        for act_num in range(num_action_space):
            ax = fig.add_subplot(1, num_action_space, act_num + 1)
            # sns.kdeplot(self.raw_actions_list[:, act_num], self.transformed_action_list[:, act_num], shade=True, ax=ax)
            ax.plot(self.raw_actions_list[:, act_num], self.transformed_action_list[:, act_num], colors[act_num])
            ax.plot([action_low, action_high], [action_low, action_high], 'k-')  # black line
            ax.plot([action_low, action_high], [0, 0], 'r-')  # red lines
            ax.plot([0, 0], [action_low, action_high], 'r-')  # red lines

            ax.title.set_text('Action Space # :' + str(act_num + 1) + '/' + str(num_action_space))
            ax.set(xlabel='Input Actions', ylabel='Transformed Actions', xlim=[action_low, action_high],
                   ylim=[action_low, action_high])
        plt.suptitle('Plot of Input action $a_t$ vs Transformed action $\\tilde{a}_t$')

        plt.savefig(expt_path)
        if show_plot: plt.show()
        plt.close()

        return mean_delta, max_delta

    def close(self):
        self.env.close()
        if self.normalizer is not None:
            self.normalizer.close()

class MujocoNormalized(gym.ObservationWrapper):
    def __init__(self, env):
        super(MujocoNormalized, self).__init__(env)
        # read this for each environment somehow
        env_name = env.spec.id
        self.max_obs = self._get_max_obs(env_name)

    def observation(self, observation):
        return observation / self.max_obs

    def _get_max_obs(self, env_name):
        # get the parent environment here
        parent_env = get_parent_env(env_name)

        max_obs_dict = {
            'InvertedPendulum-v2': np.array([0.909, 0.098, 1.078, 1.681]),
            'Hopper-v2': np.array([1.587, 0.095, 0.799, 0.191, 0.904,
                                   3.149, 2.767, 2.912, 4.063, 2.581, 10.]),
            # 'Walker2d-v2': np.array([1.547, 0.783, 0.601,0.177,1.322,0.802,0.695,1.182,4.671,3.681,
            #                            5.796,10.,10.,10.,10.,10.,10.]), # old
            'Walker2d-v2': np.array([1.35, 0.739, 1.345, 1.605, 1.387, 1.063, 1.202, 1.339, 4.988, 2.863,
                                     10., 10., 10., 10., 10., 10., 10.]),
            'MinitaurBulletEnv-v0': np.array([3.1515927, 3.1515927, 3.1515927, 3.1515927, 3.1515927,
                                              3.1515927, 3.1515927, 3.1515927, 167.72488, 167.72488,
                                              167.72488, 167.72488, 167.72488, 167.72488, 167.72488,
                                              167.72488, 5.71, 5.71, 5.71, 5.71,
                                              5.71, 5.71, 5.71, 5.71, 1.01,
                                              1.01, 1.01, 1.01]),
            'AntPyBulletEnv-v0': np.array([0.2100378, 0.5571242, 1., 1.0959914, 0.663276, 0.5758094,
                                           0.1813731, 0.2803405, 1.0526485, 1.5340704, 1.7009641, 1.6335357,
                                           1.1145028, 1.9834042, 1.6994406, 0.8969864, 1.1013167, 1.9742222,
                                           1.9255029, 0.83447146, 1.0699006, 1.5556577, 1.8345532, 1.1241446,
                                           1., 1., 1., 1.]),
            'HalfCheetah-v2': np.array([0.593, 3.618, 1.062, 0.844, 0.837, 1.088, 0.88, 0.587, 4.165, 3.58,
                                        7.851, 20.837, 25.298, 25.11, 30.665, 31.541, 15.526]),

            # 'Humanoid-v2': np.array([]),
            # 'FetchReach-v1': np.array([])
        }

        if parent_env not in max_obs_dict.keys():
            raise ValueError('Observation normalization not supported for this environment yet! .')

        return max_obs_dict[parent_env]