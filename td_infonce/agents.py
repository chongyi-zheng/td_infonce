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

"""Defines distributed TD-InfoNCE agents, using JAX."""

import functools
from typing import Callable

from acme import specs
from acme.jax import utils

from td_infonce import builder, distributed_layout
from td_infonce import networks
from td_infonce import utils as td_infonce_utils


NetworkFactory = Callable[[specs.EnvironmentSpec],
                          networks.TDInfoNCENetworks]


class DistributedTDInfoNCE(distributed_layout.DistributedLayout):
    """Distributed program definition for TD-InfoNCE."""

    def __init__(
            self,
            environment_factory,
            network_factory,
            config,
            seed,
            num_actors,
            max_number_of_steps=None,
            log_to_bigtable=False,
            log_every=10.0,
            evaluator_factories=None,
            log_dir='td_infonce_logs',
            log_dir_add_uid=False,
    ):
        # Check that the environment-specific parts of the config have been set.
        assert config.max_episode_steps > 0
        assert config.obs_dim > 0

        logger_fn = functools.partial(td_infonce_utils.make_logger,
                                      'learner', log_to_bigtable,
                                      time_delta=log_every, asynchronous=True,
                                      serialize_fn=utils.fetch_devicearray,
                                      steps_key='learner_steps',
                                      log_dir=log_dir,
                                      log_dir_add_uid=log_dir_add_uid)
        actor_logger_fn = lambda actor_id: td_infonce_utils.make_logger(
            'actor',
            save_data=(log_to_bigtable and actor_id == 0),
            time_delta=log_every,
            steps_key='actor_steps',
            log_dir=log_dir,
            log_dir_add_uid=log_dir_add_uid,
        )

        td_infonce_builder = builder.TDInfoNCEBuilder(config,
                                                      logger_fn=logger_fn)
        if evaluator_factories is None:
            eval_policy_factory = (
                lambda n: networks.apply_policy_and_sample(n, True))
            eval_observers = [td_infonce_utils.SuccessObserver()]
            eval_observers.append(
                td_infonce_utils.DistanceObserver(
                    obs_dim=config.obs_dim,
                    start_index=config.start_index,
                    end_index=config.end_index)
            )

            evaluator_logger_fn = lambda label, save_data, steps_key: td_infonce_utils.make_logger(
                label,
                save_data=save_data,
                time_delta=log_every,
                steps_key=steps_key,
                log_dir=log_dir,
                log_dir_add_uid=log_dir_add_uid
            )

            evaluator_factories = [
                distributed_layout.default_evaluator_factory(
                    environment_factory=environment_factory,
                    network_factory=network_factory,
                    policy_factory=eval_policy_factory,
                    log_to_bigtable=log_to_bigtable,
                    observers=eval_observers,
                    logger_fn=evaluator_logger_fn)
            ]
            if config.local:
                evaluator_factories = []

        actor_observers = [td_infonce_utils.SuccessObserver()]
        actor_observers.append(
            td_infonce_utils.DistanceObserver(
                obs_dim=config.obs_dim,
                start_index=config.start_index,
                end_index=config.end_index)
        )

        super().__init__(
            seed=seed,
            environment_factory=environment_factory,
            network_factory=network_factory,
            builder=td_infonce_builder,
            policy_network=networks.apply_policy_and_sample,
            evaluator_factories=evaluator_factories,
            num_actors=num_actors,
            max_number_of_steps=max_number_of_steps,
            prefetch_size=config.prefetch_size,
            log_to_bigtable=log_to_bigtable,
            actor_logger_fn=actor_logger_fn,
            observers=actor_observers,
            checkpointing_config=distributed_layout.CheckpointingConfig(
                directory=log_dir, add_uid=log_dir_add_uid),
        )
