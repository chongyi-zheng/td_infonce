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

"""Utility for loading the OpenAI Gym Fetch robotics environments."""

import gym
import gym.spaces

try:
    from gym_robotics.envs.fetch import reach

    if not hasattr(reach, 'FetchReachEnv'):
        reach.FetchReachEnv = reach.MujocoPyFetchReachEnv

    from gym_robotics.envs.fetch import push

    if not hasattr(push, 'FetchPushEnv'):
        push.FetchPushEnv = push.MujocoPyFetchPushEnv

    from gym_robotics.envs.fetch import pick_and_place

    if not hasattr(pick_and_place, 'FetchPickAndPlaceEnv'):
        pick_and_place.FetchPickAndPlaceEnv = pick_and_place.MujocoFetchPickAndPlaceEnv

    from gym_robotics.envs.fetch import slide

    if not hasattr(slide, 'FetchSlideEnv'):
        slide.FetchSlideEnv = slide.MujocoPyFetchSlideEnv


except ImportError:
    from gym.envs.robotics.fetch import reach
    from gym.envs.robotics.fetch import push
    from gym.envs.robotics.fetch import pick_and_place
    from gym.envs.robotics.fetch import slide
import numpy as np


def get_reward(norm_dist, reward_mode):
    # is_success === norm_dist < 1
    is_success = float(norm_dist < 1)
    if reward_mode == 'dense':
        return np.exp(-norm_dist * np.log(2))  # 0.5 at boundary, 1 at exact
    elif reward_mode == 'positive':
        return is_success
    else:
        assert reward_mode == 'negative'
        return is_success - 1


class FetchReachEnv(reach.FetchReachEnv):
    """Wrapper for the FetchReach environment."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        super(FetchReachEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((20,), -np.inf),
            high=np.full((20,), np.inf),
            dtype=np.float32)
        self.observation_space = self._new_observation_space

    def reset(self):
        self.observation_space = self._old_observation_space
        s = super(FetchReachEnv, self).reset()
        self.observation_space = self._new_observation_space
        return self.observation(s)

    def step(self, action):
        s = super(FetchReachEnv, self).step(action)[0]
        done = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return self.observation(s), r, done, info

    def observation(self, observation):
        start_index = 0
        end_index = 3
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)


class FetchPushEnv(push.FetchPushEnv):
    """Wrapper for the FetchPush environment."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        super(FetchPushEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((50,), -np.inf),
            high=np.full((50,), np.inf),
            dtype=np.float32)
        self.observation_space = self._new_observation_space

    def reset(self):
        self.observation_space = self._old_observation_space
        s = super(FetchPushEnv, self).reset()
        self.observation_space = self._new_observation_space
        return self.observation(s)

    def step(self, action):
        s = super(FetchPushEnv, self).step(action)[0]
        done = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return self.observation(s), r, done, info

    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[:start_index] = observation['desired_goal']
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)


class FetchPickAndPlaceEnv(pick_and_place.FetchPickAndPlaceEnv):
    """Wrapper for the FetchPush environment."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        super(FetchPickAndPlaceEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((50,), -np.inf),
            high=np.full((50,), np.inf),
            dtype=np.float32)
        self.observation_space = self._new_observation_space

    def reset(self):
        self.observation_space = self._old_observation_space
        s = super(FetchPickAndPlaceEnv, self).reset()
        self.observation_space = self._new_observation_space
        return self.observation(s)

    def step(self, action):
        s = super(FetchPickAndPlaceEnv, self).step(action)[0]
        done = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return self.observation(s), r, done, info

    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[:start_index] = observation['desired_goal']
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)


class FetchSlideEnv(slide.FetchSlideEnv):
    """Wrapper for the FetchSlide environment."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        super(FetchSlideEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((50,), -np.inf),
            high=np.full((50,), np.inf),
            dtype=np.float32)
        self.observation_space = self._new_observation_space

    def reset(self):
        self.observation_space = self._old_observation_space
        s = super(FetchSlideEnv, self).reset()
        self.observation_space = self._new_observation_space
        return self.observation(s)

    def step(self, action):
        s = super(FetchSlideEnv, self).step(action)[0]
        done = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return self.observation(s), r, done, info

    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[:start_index] = observation['desired_goal']
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)


class FetchReachImageEnv(reach.FetchReachEnv):
    """Wrapper for the FetchReach environment with image observations."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        self._dist = []
        self._dist_vec = []
        super(FetchReachImageEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((64 * 64 * 6), 0),
            high=np.full((64 * 64 * 6), 255),
            dtype=np.uint8)
        self.observation_space = self._new_observation_space
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def reset(self):
        if self._dist:  # if len(self._dist) > 0, ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        self.observation_space = self._old_observation_space
        s = super(FetchReachImageEnv, self).reset()
        self.observation_space = self._new_observation_space
        self._goal = s['desired_goal'].copy()

        for _ in range(10):
            hand = s['achieved_goal']
            obj = s['desired_goal']
            delta = obj - hand
            a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
            s = super(FetchReachImageEnv, self).step(a)[0]

        self._goal_img = self.observation(s)

        self.observation_space = self._old_observation_space
        s = super(FetchReachImageEnv, self).reset()
        self.observation_space = self._new_observation_space
        img = self.observation(s)
        dist = np.linalg.norm(s['achieved_goal'] - self._goal)
        self._dist.append(dist)
        return np.concatenate([img, self._goal_img])

    def step(self, action):
        s = super(FetchReachImageEnv, self).step(action)[0]
        dist = np.linalg.norm(s['achieved_goal'] - self._goal)
        self._dist.append(dist)
        done = False
        img = self.observation(s)
        is_success = float(dist < 0.05)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return np.concatenate([img, self._goal_img]), r, done, info

    def observation(self, observation):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode='rgb_array', height=64, width=64)
        return img.flatten()

    def _viewer_setup(self):
        super(FetchReachImageEnv, self)._viewer_setup()
        self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.5])
        self.viewer.cam.distance = 0.8
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -30

    def compute_reward(self, achieved_goal, goal, info):
        # just image comparison
        assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
        is_success = (achieved_goal == goal).all(axis=-1)
        if self.reward_mode == 'positive':
            r = is_success
        else:
            assert self.reward_mode == 'negative'
            r = is_success - 1
        return r


class FetchPushImageEnv(push.FetchPushEnv):
    """Wrapper for the FetchPush environment with image observations."""

    def __init__(self, camera='camera2', start_at_obj=True, rand_y=False,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 ):
        self.reward_mode = reward_mode
        self._start_at_obj = start_at_obj
        self._rand_y = rand_y
        self._camera_name = camera
        self._dist = []
        self._dist_vec = []
        super(FetchPushImageEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((64 * 64 * 6), 0),
            high=np.full((64 * 64 * 6), 255),
            dtype=np.uint8)
        self.observation_space = self._new_observation_space
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def _move_hand_to_obj(self):
        s = super(FetchPushImageEnv, self)._get_obs()
        for _ in range(100):
            hand = s['observation'][:3]
            obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
            delta = obj - hand
            if np.linalg.norm(delta) < 0.06:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
            s = super(FetchPushImageEnv, self).step(a)[0]

    def reset(self):
        if self._dist:  # if len(self._dist) > 0 ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        self.observation_space = self._old_observation_space
        s = super(FetchPushImageEnv, self).reset()
        self.observation_space = self._new_observation_space
        # Randomize object position
        for _ in range(8):
            super(FetchPushImageEnv, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        if not self._rand_y:
            object_qpos[1] = 0.75
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self._move_hand_to_obj()
        self._goal_img = self.observation(s)
        block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            print('Bad reset, recursing.')
            return self.reset()
        self._goal = block_xyz[:2].copy()

        self.observation_space = self._old_observation_space
        s = super(FetchPushImageEnv, self).reset()
        self.observation_space = self._new_observation_space
        for _ in range(8):
            super(FetchPushImageEnv, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:2] = np.array([1.15, 0.75])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        if self._start_at_obj:
            self._move_hand_to_obj()
        else:
            for _ in range(5):
                super(FetchPushImageEnv, self).step(self.action_space.sample())

        block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
        img = self.observation(s)
        dist = np.linalg.norm(block_xyz[:2] - self._goal)
        self._dist.append(dist)
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            print('Bad reset, recursing.')
            return self.reset()
        return np.concatenate([img, self._goal_img])

    def step(self, action):
        s = super(FetchPushImageEnv, self).step(action)[0]
        block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
        dist = np.linalg.norm(block_xy - self._goal)
        self._dist.append(dist)
        done = False
        is_success = float(dist < 0.05)  # Taken from the original task code.
        img = self.observation(s)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return np.concatenate([img, self._goal_img]), r, done, info

    def observation(self, observation):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode='rgb_array', height=64, width=64)
        return img.flatten()

    def _viewer_setup(self):
        super(FetchPushImageEnv, self)._viewer_setup()
        if self._camera_name == 'camera1':
            self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
            self.viewer.cam.distance = 0.9
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -40
        elif self._camera_name == 'camera2':
            self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
            self.viewer.cam.distance = 0.65
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -40
        else:
            raise NotImplementedError

    def compute_reward(self, achieved_goal, goal, info):
        # just image comparison
        assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
        is_success = (achieved_goal == goal).all(axis=-1)
        if self.reward_mode == 'positive':
            r = is_success
        else:
            assert self.reward_mode == 'negative'
            r = is_success - 1
        return r


class FetchSlideImageEnv(slide.FetchSlideEnv):
    """Wrapper for the FetchSlide environment with image observations."""

    def __init__(self, camera='camera2', reward_mode='positive'):
        self.reward_mode = reward_mode
        self._camera_name = camera
        self._dist = []
        self._dist_vec = []
        super(FetchSlideImageEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((64 * 64 * 6), 0),
            high=np.full((64 * 64 * 6), 255),
            dtype=np.uint8)
        self.observation_space = self._new_observation_space
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def _move_hand_to_obj(self):
        s = super(FetchSlideImageEnv, self)._get_obs()
        for _ in range(100):
            hand = s['observation'][:3]
            obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
            delta = obj - hand
            if np.linalg.norm(delta) < 0.06:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
            s = super(FetchSlideImageEnv, self).step(a)[0]

    def _raise_hand(self):
        s = super(FetchSlideImageEnv, self)._get_obs()
        for _ in range(100):
            hand = s['observation'][:3]
            target = hand + np.array([0.0, 0.0, 0.05])
            delta = target - hand
            if np.linalg.norm(delta) < 0.02:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
            s = super(FetchSlideImageEnv, self).step(a)[0]

    def reset(self):
        if self._dist:  # if len(self._dist) > 0 ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        self.observation_space = self._old_observation_space
        s = super(FetchSlideImageEnv, self).reset()
        self.observation_space = self._new_observation_space
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = self.goal
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self._raise_hand()
        self._goal_img = self.observation(s)
        block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
        self._goal = block_xyz[:2].copy()
        old_goal = self.goal.copy()

        self.observation_space = self._old_observation_space
        s = super(FetchSlideImageEnv, self).reset()
        self.observation_space = self._new_observation_space
        self._move_hand_to_obj()
        self.goal = old_goal  # set to the same goal as the goal image

        block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
        img = self.observation(s)
        dist = np.linalg.norm(block_xyz[:2] - self._goal)
        self._dist.append(dist)

        return np.concatenate([img, self._goal_img])

    def step(self, action):
        s = super(FetchSlideImageEnv, self).step(action)[0]
        block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
        dist = np.linalg.norm(block_xy - self._goal)
        self._dist.append(dist)
        done = False
        is_success = float(dist < 0.05)  # Taken from the original task code.
        img = self.observation(s)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return np.concatenate([img, self._goal_img]), r, done, info

    def observation(self, observation):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode='rgb_array', height=64, width=64)
        return img.flatten()

    def _viewer_setup(self):
        super(FetchSlideImageEnv, self)._viewer_setup()
        if self._camera_name == 'camera1':
            self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
            self.viewer.cam.distance = 0.9
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -40
        elif self._camera_name == 'camera2':
            self.viewer.cam.lookat[Ellipsis] = np.array([1.35, 0.8, 0.4])
            self.viewer.cam.distance = 1.75
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -40
        else:
            raise NotImplementedError

    def compute_reward(self, achieved_goal, goal, info):
        # just image comparison
        assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
        is_success = (achieved_goal == goal).all(axis=-1)
        if self.reward_mode == 'positive':
            r = is_success
        else:
            assert self.reward_mode == 'negative'
            r = is_success - 1
        return r


class FetchPickAndPlaceImageEnv(pick_and_place.FetchPickAndPlaceEnv):
    """Wrapper for the FetchPickAndPlace environment with image observations."""

    def __init__(self, camera='camera2', reward_mode='positive'):
        self.reward_mode = reward_mode
        self._camera_name = camera
        self._dist = []
        self._dist_vec = []
        super(FetchPickAndPlaceImageEnv, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((64 * 64 * 6), 0),
            high=np.full((64 * 64 * 6), 255),
            dtype=np.uint8)
        self.observation_space = self._new_observation_space
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def _pick_and_place(self):
        s = super(FetchPickAndPlaceImageEnv, self)._get_obs()

        # pick
        for _ in range(50):
            hand = s['observation'][:3]
            obj = s['achieved_goal'] + np.array([0.0, 0.0, 0.025])
            delta = obj - hand
            if np.linalg.norm(delta) < 0.01:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
            s = super(FetchPickAndPlaceImageEnv, self).step(a)[0]

        for _ in range(50):
            hand = s['observation'][:3]
            obj = s['achieved_goal'] + np.array([0.005, 0.0, -0.005])
            delta = obj - hand
            a = np.concatenate([np.clip(delta, -1, 1), [1.0]])
            s = super(FetchPickAndPlaceImageEnv, self).step(a)[0]

        for _ in range(30):
            a = np.concatenate([np.clip(np.zeros_like(delta), -1, 1), [-1.0]])
            s = super(FetchPickAndPlaceImageEnv, self).step(a)[0]

        # place
        for _ in range(100):
            hand = s['observation'][:3]
            goal = s['desired_goal']
            delta = goal - hand
            block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
            dist = np.linalg.norm(block_xyz - self.goal)
            is_success = float(dist < 0.05)
            if is_success:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [-1.0]])
            s = super(FetchPickAndPlaceImageEnv, self).step(a)[0]

        return is_success

    def reset(self):
        if self._dist:  # if len(self._dist) > 0 ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        is_success = False
        while not is_success:
            self.observation_space = self._old_observation_space
            s = super(FetchPickAndPlaceImageEnv, self).reset()
            self.observation_space = self._new_observation_space

            is_success = self._pick_and_place()
        self._goal_img = self.observation(s)
        old_goal = self.goal.copy()

        self.observation_space = self._old_observation_space
        s = super(FetchPickAndPlaceImageEnv, self).reset()
        self.observation_space = self._new_observation_space
        self.goal = old_goal  # set to the same goal as the goal image

        block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
        img = self.observation(s)
        dist = np.linalg.norm(block_xyz - self.goal)
        self._dist.append(dist)

        return np.concatenate([img, self._goal_img])

    def step(self, action):
        s = super(FetchPickAndPlaceImageEnv, self).step(action)[0]
        block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
        dist = np.linalg.norm(block_xyz - self.goal)
        self._dist.append(dist)
        done = False
        is_success = float(dist < 0.05)  # Taken from the original task code.
        img = self.observation(s)
        info = dict(
            is_success=is_success,
        )
        r = get_reward(dist / 0.05, self.reward_mode)
        return np.concatenate([img, self._goal_img]), r, done, info

    def observation(self, observation):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode='rgb_array', height=64, width=64)
        return img.flatten()

    def _viewer_setup(self):
        super(FetchPickAndPlaceImageEnv, self)._viewer_setup()
        if self._camera_name == 'camera1':
            self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
            self.viewer.cam.distance = 0.9
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -40
        elif self._camera_name == 'camera2':
            self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.5])
            self.viewer.cam.distance = 1.2
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30
        else:
            raise NotImplementedError

    def compute_reward(self, achieved_goal, goal, info):
        # just image comparison
        assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
        is_success = (achieved_goal == goal).all(axis=-1)
        if self.reward_mode == 'positive':
            r = is_success
        else:
            assert self.reward_mode == 'negative'
            r = is_success - 1
        return r
