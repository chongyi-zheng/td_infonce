# coding=utf-8
# Copyright 2022 The Google Research Authors.
# Copyright 2023 Chongyi Zheng.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for loading the AntMaze environments."""
import gym
import numpy as np


class OfflineD4RLWrapper(gym.ObservationWrapper):
    """Wrapper for exposing the goals of the D4RL offline environments."""

    def __init__(self, env):
        high = env.observation_space.high
        low = env.observation_space.low
        env.observation_space = gym.spaces.Box(
            low=np.full((low.shape[0] * 2,), -np.inf),
            high=np.full((high.shape[0] * 2,), np.inf),
            dtype=np.float32,
        )
        super(OfflineD4RLWrapper, self).__init__(env)

    def observation(self, observation):
        goal_obs = np.zeros_like(observation)
        if hasattr(self.env, 'target_goal'):
            goal_obs[:2] = self.env.target_goal
        elif hasattr(self.env, 'goal_locations'):
            goal_obs[:2] = self.env.goal_locations[0]
        else:
            raise NotImplementedError
        return np.concatenate([observation, goal_obs])

    @property
    def max_episode_steps(self):
        if hasattr(self.env, 'max_episode_steps'):
            return self.env.max_episode_steps
        elif hasattr(self.env, '_max_episode_steps'):
            return self.env._max_episode_steps
        else:
            raise NotImplementedError


def make_offline_d4rl(env_name):
    """Loads the D4RL AntMaze environments."""
    if env_name == 'offline_ant_umaze':
        env = gym.make('antmaze-umaze-v2')
    elif env_name == 'offline_ant_umaze_diverse':
        env = gym.make('antmaze-umaze-diverse-v2')
    elif env_name == 'offline_ant_medium_play':
        env = gym.make('antmaze-medium-play-v2')
    elif env_name == 'offline_ant_medium_diverse':
        env = gym.make('antmaze-medium-diverse-v2')
    elif env_name == 'offline_ant_large_play':
        env = gym.make('antmaze-large-play-v2')
    elif env_name == 'offline_ant_large_diverse':
        env = gym.make('antmaze-large-diverse-v2')
    else:
        raise NotImplementedError

    return OfflineD4RLWrapper(env.env)
