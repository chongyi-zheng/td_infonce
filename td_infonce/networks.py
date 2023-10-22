# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Copyright Chongyi Zheng.
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

"""TD-InfoNCE networks definition."""
import dataclasses
from typing import Optional, Callable

from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class TDInfoNCENetworks:
    """Network and pure functions for the TD-InfoNCE agent."""
    policy_network: networks_lib.FeedForwardNetwork
    q_network: networks_lib.FeedForwardNetwork
    log_prob: networks_lib.LogProbFn
    repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
    sample: networks_lib.SampleFn
    sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(
        networks,
        eval_mode=False):
    """Returns a function that computes actions."""
    sample_fn = networks.sample if not eval_mode else networks.sample_eval
    if not sample_fn:
        raise ValueError('sample function is not provided')

    def apply_and_sample(params, key, obs):
        return sample_fn(networks.policy_network.apply(params, obs), key)

    return apply_and_sample


def make_networks(
        spec,
        obs_dim,
        repr_dim=64,
        repr_norm=False,
        repr_norm_temp=1.0,
        hidden_layer_sizes=(256, 256),
        actor_min_std=1e-6,
        twin_q=False,
        use_image_obs=False):
    """Creates networks used by the agent."""

    num_dimensions = np.prod(spec.actions.shape, dtype=int)
    TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

    def _unflatten_img(img):
        img = jnp.reshape(img, (-1, 64, 64, 3)) / 255.0
        return img

    def _repr_fn(obs, action, goal, future_obs):
        # The optional input hidden is the image representations. We include this
        # as an input for the second Q value when twin_q = True, so that the two Q
        # values use the same underlying image representation.
        if use_image_obs:
            obs = _unflatten_img(obs)
            goal = _unflatten_img(goal)
            future_obs = _unflatten_img(future_obs)
            img_encoder = TORSO()
            state = img_encoder(obs)
            goal = img_encoder(goal)
            future_state = img_encoder(future_obs)
        else:
            state = obs
            goal = goal
            future_state = future_obs

        sag_encoder = hk.Sequential([
            hk.nets.MLP(
                list(hidden_layer_sizes) + [repr_dim],
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu,
                name='sag_encoder'),
        ])
        sag_repr = sag_encoder(jnp.concatenate([state, action, goal], axis=-1))

        sag_encoder2 = hk.Sequential([
            hk.nets.MLP(
                list(hidden_layer_sizes) + [repr_dim],
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu,
                name='sag_encoder2'),
        ])
        sag_repr2 = sag_encoder2(jnp.concatenate([state, action, goal], axis=-1))
        sag_repr = jnp.stack([sag_repr, sag_repr2], axis=-1)

        fs_encoder = hk.Sequential([
            hk.nets.MLP(
                list(hidden_layer_sizes) + [repr_dim],
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu,
                name='fs_encoder'),
        ])
        fs_repr = fs_encoder(future_state)

        fs_encoder2 = hk.Sequential([
            hk.nets.MLP(
                list(hidden_layer_sizes) + [repr_dim],
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu,
                name='fs_encoder2'),
        ])
        fs_repr2 = fs_encoder2(future_state)
        fs_repr = jnp.stack([fs_repr, fs_repr2], axis=-1)

        if repr_norm:
            sag_repr = sag_repr / (jnp.linalg.norm(sag_repr, axis=1, keepdims=True) + 1e-8)
            fs_repr = fs_repr / (jnp.linalg.norm(fs_repr, axis=1, keepdims=True) + 1e-8)

            sag_repr = sag_repr / repr_norm_temp

        return sag_repr, fs_repr

    def _combine_repr(sag_repr, fs_repr):
        return jnp.einsum('ikl,jkl->ijl', sag_repr, fs_repr)

    def _critic_fn(obs, action, goal, future_obs, repr=False):
        assert twin_q
        sag_repr, fs_repr = _repr_fn(
            obs, action, goal, future_obs)
        outer = _combine_repr(sag_repr, fs_repr)

        if repr:
            return outer, sag_repr, fs_repr
        else:
            return outer

    def _actor_fn(obs_and_goal):
        if use_image_obs:
            obs, goal = obs_and_goal[:, :obs_dim], obs_and_goal[:, obs_dim:]
            obs = _unflatten_img(obs)
            goal = _unflatten_img(goal)
            img_encoder = TORSO()
            state_and_goal = img_encoder(jnp.concatenate([obs, goal], axis=-1))
        else:
            state_and_goal = obs_and_goal
        network = hk.Sequential([
            hk.nets.MLP(
                list(hidden_layer_sizes),
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu,
                activate_final=True),
            networks_lib.NormalTanhDistribution(num_dimensions,
                                                min_scale=actor_min_std),
        ])

        return network(state_and_goal)

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))
    repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))

    # Create dummy observations and actions to create network parameters.
    dummy_action = utils.ones_like(spec.actions)
    dummy_obs = utils.ones_like(spec.observations)[:obs_dim]
    dummy_future_obs = utils.ones_like(spec.observations)[obs_dim:]
    dummy_goal = utils.ones_like(spec.observations)[obs_dim:]
    dummy_obs_and_goal = utils.ones_like(spec.observations)
    dummy_action = utils.add_batch_dim(dummy_action)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    dummy_future_obs = utils.add_batch_dim(dummy_future_obs)
    dummy_goal = utils.add_batch_dim(dummy_goal)
    dummy_obs_and_goal = utils.add_batch_dim(dummy_obs_and_goal)

    return TDInfoNCENetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs_and_goal),
            policy.apply
        ),
        q_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action, dummy_goal, dummy_future_obs),
            critic.apply
        ),
        repr_fn=repr_fn.apply,
        log_prob=lambda params, actions: params.log_prob(actions),
        sample=lambda params, key: params.sample(seed=key),
        sample_eval=lambda params, key: params.mode(),
    )
