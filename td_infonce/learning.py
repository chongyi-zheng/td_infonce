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

"""TD-InfoNCE learner implementation."""
import time
from typing import NamedTuple, Optional

import acme
import jax
import jax.numpy as jnp
import optax
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers


class TrainingState(NamedTuple):
    """Contains training state for the learner."""
    policy_optimizer_state: optax.OptState
    q_optimizer_state: optax.OptState
    policy_params: networks_lib.Params
    q_params: networks_lib.Params
    target_q_params: networks_lib.Params
    key: networks_lib.PRNGKey


class TDInfoNCELearner(acme.Learner):
    """TD-InfoNCE learner."""

    _state: TrainingState

    def __init__(
            self,
            networks,
            rng,
            policy_optimizer,
            q_optimizer,
            iterator,
            counter,
            logger,
            obs_to_goal,
            config):
        """Initialize the TD-InfoNCE learner.

        Args:
          networks: TD-InfoNCE networks.
          rng: a key for random number generation.
          policy_optimizer: the policy optimizer.
          q_optimizer: the Q-function optimizer.
          iterator: an iterator over training data.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          obs_to_goal: a function for extracting the goal coordinates.
          config: the experiment config file.
        """
        self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
        self._obs_dim = config.obs_dim

        def critic_loss(q_params,
                        policy_params,
                        target_q_params,
                        transitions,
                        key):
            batch_size = transitions.observation.shape[0]

            obs, _ = jnp.split(transitions.observation, [config.obs_dim], axis=1)
            goal = transitions.extras['random_goal']
            action = transitions.action
            next_obs, _ = jnp.split(transitions.next_observation, [config.obs_dim], axis=1)
            rand_g = jnp.roll(goal, -1, axis=0)

            # term 1
            pos_logits = networks.q_network.apply(
                q_params, obs, action, goal, obs_to_goal(next_obs))
            I = jnp.eye(batch_size)
            I = I[:, :, None].repeat(pos_logits.shape[-1], axis=-1)
            loss1 = jax.vmap(optax.softmax_cross_entropy, -1, -1)(pos_logits, I)

            # term 2
            next_dist_params = networks.policy_network.apply(
                policy_params, jnp.concatenate([next_obs, goal], axis=-1))
            next_action = networks.sample(next_dist_params, key)

            neg_logits = networks.q_network.apply(
                q_params, obs, action, goal, rand_g)

            # importance sampling weight
            logits_w = networks.q_network.apply(
                target_q_params, next_obs, next_action, goal, rand_g)

            logits_w = jnp.min(logits_w, axis=-1)
            w = jax.nn.softmax(logits_w, axis=1)

            # Note that we remove the multiplier N for w to balance
            # one term of loss1 with N terms of loss2 in each row.
            w = jax.lax.stop_gradient(w)  # (N, N)
            loss2 = jax.vmap(optax.softmax_cross_entropy, -1, -1)(
                neg_logits,
                w[:, :, None].repeat(neg_logits.shape[-1], axis=-1)
            )

            loss = (1 - config.discount) * loss1 + config.discount * loss2
            loss = jnp.mean(loss)

            logits_pos_entropy = -jnp.sum(
                jax.nn.softmax(pos_logits, axis=1) * jax.nn.log_softmax(pos_logits, axis=1), axis=1)
            logits_neg_entropy = -jnp.sum(
                jax.nn.softmax(neg_logits, axis=1) * jax.nn.log_softmax(neg_logits, axis=1), axis=1)
            logits_w_entropy = -jnp.sum(
                jax.nn.softmax(logits_w, axis=1) * jax.nn.log_softmax(logits_w, axis=1), axis=1)

            metrics = {
                "loss1": jnp.mean(loss1),
                "loss2": jnp.mean(loss2),
                "logits_pos": jnp.mean(jax.vmap(jnp.diag, -1, -1)(pos_logits)),
                "logits_pos1": jnp.mean(jnp.diag(pos_logits[..., 0])),
                "logits_pos2": jnp.mean(jnp.diag(pos_logits[..., 1])),
                "logits_pos_entropy": jnp.mean(logits_pos_entropy),
                "logits_neg": jnp.mean(neg_logits),
                "logits_neg1": jnp.mean(neg_logits[..., 0]),
                "logits_neg2": jnp.mean(neg_logits[..., 1]),
                "logits_neg_entropy": jnp.mean(logits_neg_entropy),
                "w_diag": jnp.mean(jnp.diag(w)),
                "w": jnp.mean(w),
                "logits_w_entropy": jnp.mean(logits_w_entropy),
            }

            return loss, metrics

        def actor_loss(policy_params,
                       q_params,
                       transitions,
                       key,
                       ):
            obs_and_goal = transitions.observation
            if config.use_gcbc:
                dist_params = networks.policy_network.apply(
                    policy_params, obs_and_goal)
                log_prob = networks.log_prob(dist_params, transitions.action)
                actor_loss = -1.0 * jnp.mean(log_prob)
            else:
                obs = obs_and_goal[:, :config.obs_dim]
                future_goal = transitions.extras['future_goal']
                random_goal = transitions.extras['random_goal']

                if config.random_goals == 0.0:
                    new_obs = obs
                    new_goal = future_goal
                elif config.random_goals == 0.5:
                    new_obs = jnp.concatenate([obs, obs], axis=0)
                    new_goal = jnp.concatenate([future_goal, random_goal], axis=0)
                else:
                    assert config.random_goals == 1.0
                    new_obs = obs
                    new_goal = random_goal

                new_obs_and_goal = jnp.concatenate([new_obs, new_goal], axis=1)
                batch_size = new_obs_and_goal.shape[0]

                dist_params = networks.policy_network.apply(
                    policy_params, new_obs_and_goal)
                action = networks.sample(dist_params, key)
                logits = networks.q_network.apply(
                    q_params, new_obs, action, new_goal, new_goal)

                logits = jnp.min(logits, axis=-1)
                I = jnp.eye(batch_size)
                actor_q_loss = optax.softmax_cross_entropy(logits=logits, labels=I)
                actor_loss = actor_q_loss

                assert 0.0 <= config.bc_coef <= 1.0
                if config.bc_coef > 0:
                    orig_action = transitions.action

                    train_mask = jnp.float32((orig_action * 1E8 % 10)[:, 0] != 4)
                    val_mask = 1.0 - train_mask

                    # MSE bc loss
                    bc_loss = train_mask * jnp.mean((action - orig_action) ** 2, axis=1)
                    bc_val_loss = val_mask * jnp.mean((action - orig_action) ** 2, axis=1)

                    actor_loss = config.bc_coef * bc_loss + (1 - config.bc_coef) * actor_loss
                else:
                    bc_loss = 0.0
                    bc_val_loss = 0.0
                    train_mask = 1.0
                    val_mask = 0.0

                actor_loss = jnp.mean(actor_loss)

            metrics = {
                "gcbc_loss": jnp.sum(bc_loss) / (jnp.sum(train_mask) + 1e-8),
                "gcbc_val_loss": jnp.sum(bc_val_loss) / (jnp.sum(val_mask) + 1e-8),
                "actor_q_loss": jnp.mean(actor_q_loss),
            }

            return actor_loss, metrics

        critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
        actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

        def update_step(
            state,
            transitions
        ):
            key_critic, key_actor, key = jax.random.split(state.key, 3)

            if not config.use_gcbc:
                (critic_loss, critic_metrics), critic_grads = critic_grad(
                    state.q_params, state.policy_params, state.target_q_params,
                    transitions, key_critic)

            (actor_loss, actor_metrics), actor_grads = actor_grad(
                state.policy_params, state.q_params, transitions, key_actor)

            # Apply policy gradients
            actor_update, policy_optimizer_state = policy_optimizer.update(
                actor_grads, state.policy_optimizer_state)
            policy_params = optax.apply_updates(state.policy_params, actor_update)

            # compute actor gradient norm
            actor_grad_norm = 0
            for g in jax.tree_leaves(actor_grads):
                actor_grad_norm += (jnp.linalg.norm(g) ** 2)
            actor_grad_norm = jnp.sqrt(actor_grad_norm)

            # Apply critic gradients
            if config.use_gcbc:
                metrics = {}
                critic_loss = 0.0
                q_params = state.q_params
                q_optimizer_state = state.q_optimizer_state
                new_target_q_params = state.target_q_params
                critic_grad_norm = 0
            else:
                critic_update, q_optimizer_state = q_optimizer.update(
                    critic_grads, state.q_optimizer_state)

                q_params = optax.apply_updates(state.q_params, critic_update)

                new_target_q_params = jax.tree_map(
                    lambda x, y: x * (1 - config.tau) + y * config.tau,
                    state.target_q_params, q_params)
                metrics = critic_metrics

                # compute critic gradient norm
                critic_grad_norm = 0
                for g in jax.tree_leaves(critic_grads):
                    critic_grad_norm += (jnp.linalg.norm(g) ** 2)
                critic_grad_norm = jnp.sqrt(critic_grad_norm)

            metrics.update(actor_metrics)

            metrics.update({
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'critic_grad_norm': critic_grad_norm,
                'actor_grad_norm': actor_grad_norm,
            })

            new_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                q_optimizer_state=q_optimizer_state,
                policy_params=policy_params,
                q_params=q_params,
                target_q_params=new_target_q_params,
                key=key,
            )

            return new_state, metrics

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        # Dummy increment to prevent field missing in the evaluator CSV log.
        self._counter.increment(steps=0, walltime=0)
        self._logger = logger or loggers.make_default_logger(
            'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray,
            time_delta=10.0)

        # Iterator on demonstration transitions.
        self._iterator = iterator

        update_step = utils.process_multiple_batches(update_step,
                                                     config.num_sgd_steps_per_step)
        # Use the JIT compiler.
        if config.jit:
            self._update_step = jax.jit(update_step)
        else:
            self._update_step = update_step

        def make_initial_state(key):
            """Initialises the training state (parameters and optimiser state)."""
            key_policy, key_q, key = jax.random.split(key, 3)

            policy_params = networks.policy_network.init(key_policy)
            policy_optimizer_state = policy_optimizer.init(policy_params)

            q_params = networks.q_network.init(key_q)
            q_optimizer_state = q_optimizer.init(q_params)

            state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                q_optimizer_state=q_optimizer_state,
                policy_params=policy_params,
                q_params=q_params,
                target_q_params=q_params,
                key=key)

            return state

        # Create initial state.
        self._state = make_initial_state(rng)

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online
        # and fill the replay buffer.
        self._timestamp = None

    def step(self):
        with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
            sample = next(self._iterator)
            transitions = types.Transition(*sample.data)
            self._state, metrics = self._update_step(self._state, transitions)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Increment counts and record the current time
        counts = self._counter.increment(
            steps=self._num_sgd_steps_per_step, walltime=elapsed_time)
        if elapsed_time > 0:
            metrics['steps_per_second'] = (
                    self._num_sgd_steps_per_step / elapsed_time)
        else:
            metrics['steps_per_second'] = 0.

        # Attempts to write the logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names):
        variables = {
            'policy': self._state.policy_params,
            'critic': self._state.q_params,
        }
        return [variables[name] for name in names]

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state
