from collections import OrderedDict
import numpy as np
from typing import Sequence

from stable_baselines.common.vec_env.base_vec_env import VecEnv
from stable_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``, as the overhead of
    multiprocess or multithread outweighs the environment computation time. This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
            for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def seed(self, seed=None):
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        return [env.render(*args, mode='rgb_array', **kwargs) for env in self.envs]

    def render(self, *args, **kwargs):
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via `BaseVecEnv.render()`.
        Otherwise (if `self.num_envs == 1`), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as `mode` will have values that are valid
        only when `num_envs == 1`.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:

                # print(obs.shape)

                from itertools import chain
                act = obs[1:].reshape(1, -1)
                obs = np.asarray([list(chain.from_iterable(obs[0].values()))])

                # print(obs.shape)
                # print(act.shape)
                
                obs = np.concatenate((obs,act), axis=1).astype('float32').reshape(-1)

                # print(obs.shape)
                # print(self.buf_obs[key][env_idx].shape)
                # print(self.buf_obs[key][env_idx])
                # print(temp.shape)
                
                # print(self.buf_obs[key][env_idx].shape)
                # quit()

                # l = obs.shape[1]
                # (self.buf_obs[key][env_idx])[:, :l] = obs

                # self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
