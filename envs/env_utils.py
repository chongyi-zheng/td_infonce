# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Utility for loading the goal-conditioned environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from envs import offline_ant_envs
from envs import fetch_envs

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def load(env_name):
    """Loads the train and eval environments, as well as the obs_dim."""
    # pylint: disable=invalid-name
    kwargs = {}
    if env_name == 'fetch_reach':
        CLASS = fetch_envs.FetchReachEnv
        max_episode_steps = 50
    elif env_name == 'fetch_push':
        CLASS = fetch_envs.FetchPushEnv
        max_episode_steps = 50
    elif env_name == 'fetch_reach_image':
        CLASS = fetch_envs.FetchReachImageEnv
        max_episode_steps = 50
    elif env_name == 'fetch_push_image':
        CLASS = fetch_envs.FetchPushImageEnv
        max_episode_steps = 50
    elif env_name == 'fetch_slide':
        CLASS = fetch_envs.FetchSlideEnv
        max_episode_steps = 50
    elif env_name == 'fetch_slide_image':
        CLASS = fetch_envs.FetchSlideImageEnv
        max_episode_steps = 50
    elif env_name == 'fetch_pick_and_place':
        CLASS = fetch_envs.FetchPickAndPlaceEnv
        max_episode_steps = 50
    elif env_name == 'fetch_pick_and_place_image':
        CLASS = fetch_envs.FetchPickAndPlaceImageEnv
        max_episode_steps = 50
    elif env_name.startswith('offline_ant'):
        CLASS = lambda: offline_ant_envs.make_offline_d4rl(env_name)
        if 'umaze' in env_name:
            max_episode_steps = 700
        else:
            max_episode_steps = 1000
    else:
        raise NotImplementedError('Unsupported environment: %s' % env_name)

    # Disable type checking in line below because different environments have
    # different kwargs, which pytype doesn't reason about.
    gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
    obs_dim = gym_env.observation_space.shape[0] // 2
    return gym_env, obs_dim, max_episode_steps
