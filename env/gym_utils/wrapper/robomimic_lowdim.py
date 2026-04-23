"""
Environment wrapper for Robomimic environments with state observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py

For consistency, we will use Dict{} for the observation space, with the key "state" for the state observation.
"""

import numpy as np
import gym
from gym import spaces
import imageio


class RobomimicLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
    ):
        self.env = env
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1.0, dtype=np.float32)
        high = np.full(env.action_dimension, fill_value=1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )
        self.obs_keys = low_dim_keys
        self.observation_space = spaces.Dict()
        obs_example_full = self.env.get_observation()
        obs_example = np.concatenate(
            [self._flatten_obs_value(obs_example_full[key]) for key in self.obs_keys], axis=0
        )
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def _prepare_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        expected_shape = self.action_space.shape
        if action.shape != expected_shape:
            try:
                action = action.reshape(expected_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Expected action shape {expected_shape}, got {action.shape}"
                ) from exc
        if not np.isfinite(action).all():
            raise ValueError(
                "RobomimicLowdimWrapper received non-finite action values: "
                f"min={np.nanmin(action)!r}, max={np.nanmax(action)!r}"
            )
        if self.normalize:
            action = np.clip(action, -1.0, 1.0)
            action = self.unnormalize_action(action)
        return np.ascontiguousarray(action, dtype=np.float32)

    @staticmethod
    def _flatten_obs_value(value):
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            return array.reshape(1)
        return array.reshape(-1)

    def get_observation(self, raw_obs):
        obs = {
            "state": np.concatenate(
                [self._flatten_obs_value(raw_obs[key]) for key in self.obs_keys],
                axis=0,
            )
        }
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        action = self._prepare_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        info = dict(info)
        success_fn = getattr(self.env, "is_success", None)
        if callable(success_fn):
            try:
                info["success"] = bool(success_fn().get("task", False))
            except Exception:
                info["success"] = False
        obs = self.get_observation(raw_obs)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )
