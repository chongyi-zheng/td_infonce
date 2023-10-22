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

# python3
r"""Example running TD-InfoNCE in JAX.

Run using multi-processing (required for image-based experiments):
  python lp_td_infonce.py --lp_launch_type=local_mp

Run using multi-threading
  python lp_td_infonce.py --lp_launch_type=local_mt
"""
import os
import functools
import logging

import tensorflow as tf
import launchpad as lp
from absl import app
from absl import flags

from td_infonce.agents import DistributedTDInfoNCE
from td_infonce.config import TDInfoNCEConfig
from td_infonce.networks import make_networks
from td_infonce.utils import make_environment

# disable tensorflow_probability warning: The use of `check_types` is deprecated and does not have any effect.
logger = logging.getLogger("root")


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'Runs training for just a few steps.')
flags.DEFINE_bool('run_tf_eagerly', False, 'Enables / disables eager execution of tf.functions.')
flags.DEFINE_string('exp_log_dir', os.path.join(os.path.realpath(__file__), 'td_infonce_logs'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_bool('exp_log_dir_add_uid', False,
                  'Enables / disables unique id for the log directory')
flags.DEFINE_string('env_name', 'sawyer_window',
                    'Select an environment')
# fetch_reach / fetch_reach_image and offline antmaze tasks: max_number_of_steps = 500_000
# other fetch tasks: max_number_of_steps = 1_000_000
flags.DEFINE_integer('max_number_of_steps', 500_000,
                     'For online RL experiments, max_number_of_steps is the number of '
                     'environment steps. For offline RL experiments, this is the number of'
                     'gradient steps.')
flags.DEFINE_bool('jit', True,
                  'Enables / disables jax.jit compilation')
flags.DEFINE_integer('seed', 0, 'Random seed')


@functools.lru_cache()
def get_env(env_name, start_index, end_index):
    return make_environment(env_name, start_index, end_index, seed=0)


def get_program(params):
    """Constructs the program."""

    env_name = params['env_name']
    seed = params.pop('seed')

    if params.get('use_image_obs', False) and not params.get('local', False):
        print('WARNING: overwriting parameters for image-based tasks.')
        params['num_sgd_steps_per_step'] = 8
        params['prefetch_size'] = 8
        params['num_actors'] = 5

    if env_name.startswith('offline'):
        # No actors needed for the offline RL experiments. Evaluation is
        # handled separately.
        params['num_actors'] = 0

    config = TDInfoNCEConfig(**params)

    env_factory = lambda seed: make_environment(  # pylint: disable=g-long-lambda
        env_name, config.start_index, config.end_index, seed)

    env_factory_no_extra = lambda seed: env_factory(seed)[0]  # Remove obs_dim.
    environment, obs_dim = get_env(env_name, config.start_index,
                                   config.end_index)
    assert (environment.action_spec().minimum == -1).all()
    assert (environment.action_spec().maximum == 1).all()
    config.obs_dim = obs_dim
    config.max_episode_steps = getattr(environment, '_step_limit')
    if env_name == 'offline_ant_umaze_diverse':
        # This environment terminates after 700 steps, but demos have 1000 steps.
        config.max_episode_steps = 1000

    network_factory = functools.partial(
        make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
        repr_norm=config.repr_norm, repr_norm_temp=config.repr_norm_temp,
        twin_q=config.twin_q,
        use_image_obs=config.use_image_obs,
        hidden_layer_sizes=config.hidden_layer_sizes)

    agent = DistributedTDInfoNCE(
        seed=seed,
        environment_factory=env_factory_no_extra,
        network_factory=network_factory,
        config=config,
        num_actors=config.num_actors,
        log_to_bigtable=True,
        max_number_of_steps=config.max_number_of_steps,
        log_dir=FLAGS.exp_log_dir,
        log_dir_add_uid=FLAGS.exp_log_dir_add_uid)
    return agent.build()


def main(_):
    if FLAGS.run_tf_eagerly:
        tf.config.run_functions_eagerly(True)  # debug tensorflow functions

    # 1. Select an environment.
    # Supported environments:
    #   OpenAI Gym Fetch: fetch_{reach,push,pick_and_place,slide}
    # Image observation environments:
    #   OpenAI Gym Fetch: fetch_{reach,push,pick_and_place,slide}_image
    # Offline environments:
    #   antmaze: offline_ant_{umaze,umaze_diverse,
    #                         medium_play,medium_diverse,
    #                         large_play,large_diverse}
    env_name = FLAGS.env_name
    params = {
        'seed': FLAGS.seed,
        'jit': FLAGS.jit,
        'use_random_actor': True,
        'env_name': env_name,
        # For online RL experiments, max_number_of_steps is the number of
        # environment steps. For offline RL experiments, this is the number of
        # gradient steps.
        'max_number_of_steps': FLAGS.max_number_of_steps,
        'use_image_obs': 'image' in env_name,
    }
    if 'ant_' in env_name or 'maze2d_' in env_name or 'point_' in env_name:
        params['end_index'] = 2

    # For the offline RL experiments, modify some hyperparameters.
    if env_name.startswith('offline'):
        params.update({
            # Effectively remove the rate-limiter by using very large values.
            'samples_per_insert': 1_000_000,
            'samples_per_insert_tolerance_rate': 100_000_000.0,
            # For the actor update, only use future states as goals.
            'random_goals': 0.0,
            'bc_coef': 0.2,  # Add a behavioral cloning term to the actor. AntMazeLarge MSE loss.
            'batch_size': 1024,  # Increase the batch size 256 --> 1024.
            # Increase the policy network size (512, 512, 512, 512) --> (1024, 1024, 1024, 1024)
            'hidden_layer_sizes': (1024, 1024, 1024, 1024),  # AntMaze doesn't seem to need a large network,
        })

    # 2. Select compute parameters. The default parameters are already tuned, so
    # use this mainly for debugging.
    if FLAGS.debug:
        params.update({
            'min_replay_size': 4_000,
            'local': True,
            'num_sgd_steps_per_step': 1,
            'prefetch_size': 1,
            'num_actors': 1,
            'batch_size': 32,
            'max_number_of_steps': 20_000,
            'hidden_layer_sizes': (32, 32, 32, 32),
            'bc_early_stopping_steps': 2_000,
            'twin_q': True,
        })

    program = get_program(params)
    # Set terminal='tmux' if you want different components in different windows.
    lp.launch(program, terminal='current_terminal')


if __name__ == '__main__':
    app.run(main)
