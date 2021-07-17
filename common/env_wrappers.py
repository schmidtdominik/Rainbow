"""
Here all environment wrappers are defined and environments are created and configured
"""

import time
from functools import partial

import cv2, imageio
import gym, retro
from retro.examples.discretizer import Discretizer
import numpy as np

import common.retro_utils as retro_utils
from common.vec_envs import SubprocVecEnvNoFlatten, DummyVecEnvNoFlatten, LazyVecFrameStack

cv2.ocl.setUseOpenCL(False)

EMULATOR_REC_SCALE = 2
PREPROC_REC_SCALE = 4
BASE_FPS_ATARI = 100
BASE_FPS_PROCGEN = 22

# Some of these wrappers are based on https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.t0 = time.time()
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(self, **kwargs):
        observation = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return observation

    def step(self, action):
        observation, reward, done, info = super(RecordEpisodeStatistics, self).step(action)
        self.episode_return += reward
        self.episode_length += 1
        if done:
            info['episode_metrics'] = {'return': self.episode_return,
                                       'length': self.episode_length,
                                       'time': round(time.time() - self.t0, 6)}

            self.episode_return = 0.0
            self.episode_length = 0
            self.t0 = time.time()
        return observation, reward, done, info


class TimeLimit(gym.Wrapper):
    # this TimeLimit is slightly different than the one in gym.wrappers. This one is from:
    # (from https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/wrappers.py)

    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
            print('Truncated episode due to time limit!')
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, seed, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        np.random.seed(seed)
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class RetroEpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.enabled = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not self.enabled:
            return obs, reward, done, info

        if self.enabled and not 'lives' in info:
            self.enabled = False
            print('Warning: RetroEpisodicLifeEnv wrapper was disabled since this env does not seem to track player lives.')
            return obs, reward, done, info

        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info['lives']
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if not self.enabled:
            return self.env.reset(**kwargs)

        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.lives = info['lives']
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        actual_rewards = []
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            actual_rewards.append(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        info['actual_rewards'] = actual_rewards
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SkipFrameEnv(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame without maxing consecutive frames"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        actual_rewards = []

        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            actual_rewards.append(reward)
            if done:
                break
        info['actual_rewards'] = actual_rewards
        return obs, total_reward, done, info


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob, seed):
        print(stickprob)
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState(seed)
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        actual_rewards = []
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            actual_rewards.append(rew)

            if done: break

        info['actual_rewards'] = actual_rewards
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)


class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env, frame_skip_aware=False):
        super().__init__(env)
        # only makes a difference when there a multiple nonzero rewards within the frame skip window
        self.frame_skip_aware = frame_skip_aware

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if not self.frame_skip_aware:
            reward = np.sign(reward)
        else:
            assert 'actual_rewards' in info
            reward = np.sign(info['actual_rewards']).mean() * 4

        return obs, reward.astype(np.float32), done, info


class RecorderWrapper(gym.Wrapper):
    """ Env wrapper that records the game as an .mp4 """

    def __init__(self, env, fps, save_dir, label, record_every):
        super().__init__(env)
        self.record_every = record_every
        self.save_dir = save_dir
        self.label = label
        assert self.label in ('emulator', 'preproc')
        self.fps = fps
        self.recordings = 0
        self.writer = None
        self.frames_written = 0

        self.last_recording = 0

        self.scale_factor = None

    def step(self, action):
        observation, rew, done, info = self.env.step(action)

        if done and self.is_recording:
            self.writer.close()
            self.writer = None
            self.frames_written = 0
            self.last_recording = time.time()
            info[self.label + '_recording'] = self.save_dir + f'/{self.label}_{self.recordings}.mp4'
            self.recordings += 1

        if time.time() - self.last_recording > self.record_every and not self.is_recording and done:
            self.frames_written = 0
            self.writer = imageio.get_writer(self.save_dir + f'/{self.label}_{self.recordings}.mp4', fps=self.fps, macro_block_size=1)
            self.last_recording = time.time()

        if self.writer is not None and self.frames_written < (60 * 60 * 16 if self.label == 'preproc' else 60 * 60 * 9):
            if self.scale_factor is None:
                self.scale_factor = EMULATOR_REC_SCALE if self.label == 'emulator' else PREPROC_REC_SCALE
                if observation.shape[0] <= 64 and observation.shape[1] <= 64:
                    self.scale_factor *= 2

            rec_observation = cv2.resize(observation, (observation.shape[1] * self.scale_factor, observation.shape[0] * self.scale_factor),
                                         interpolation=cv2.INTER_NEAREST)
            self.frames_written += 1
            self.writer.append_data(rec_observation.squeeze())
        return observation, rew, done, info

    @property
    def is_recording(self):
        return self.writer is not None


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height, grayscale=True, interp=cv2.INTER_AREA, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self.interp = interp
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if frame.shape[0] != self._height or frame.shape[1] != self._width:  # ds maybe

            if (self._height != self._width and frame.shape[0] != frame.shape[1]) and ((frame.shape[0] > frame.shape[1]) != (self._height > self._width)):
                frame = frame.transpose(1, 0, 2)
            frame = cv2.resize(frame, (self._width, self._height), interpolation=self.interp)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class RandomizeStateOnReset(gym.Wrapper):

    def __init__(self, env, seed):
        super().__init__(env)
        self.init_states = retro_utils.get_init_states()[self.env.gamename]
        print(self.init_states)
        self.rng = np.random.RandomState(seed)
        if self.init_states:
            self.unwrapped.load_state(self.init_states[self.rng.randint(0, len(self.init_states))])

    def reset(self, *args, **kwargs):
        if len(self.init_states) > 1:
            next_state = self.init_states[self.rng.randint(0, len(self.init_states))]
            print(f'Loading state {next_state}')
            self.unwrapped.load_state(next_state)
        return self.env.reset(*args, **kwargs)


class DecorrEnvWrapper(gym.Wrapper):

    def __init__(self, env, decorr_steps):
        super().__init__(env)
        self.decorr_steps = decorr_steps
        self.done = False

    def reset(self):
        state = self.env.reset()

        if not self.done:
            for i in range(int(self.decorr_steps)):
                state, _, _, _ = self.env.step(self.env.action_space.sample())
            self.done = True
            print(self.decorr_steps)
        return state


def create_atari_env(config, instance_seed, instance, decorr_steps):
    """ Configure environment for DeepMind-style Atari """

    env = gym.make(config.env_name[4:] + 'NoFrameskip-v4')
    env = TimeLimit(env, config.time_limit)

    if instance == 0:
        env = RecorderWrapper(env, fps=BASE_FPS_ATARI, save_dir=config.save_dir, label='emulator', record_every=config.record_every)

    env = NoopResetEnv(env, instance_seed, noop_max=30)
    env = MaxAndSkipEnv(env, skip=config.frame_skip)

    env = RecordEpisodeStatistics(env)  # this has to be applied before the EpisodicLifeEnv wrapper!

    # NOTE: it's unclear whether using the EpisodicLifeEnv and FireResetEnv wrappers yield any benefits
    # see https://github.com/openai/baselines/issues/240#issuecomment-391165056
    # and https://github.com/astooke/rlpyt/pull/158#issuecomment-632859702
    env = EpisodicLifeEnv(env)
    # if 'FIRE' in env.unwrapped.get_action_meanings():
    #    env = FireResetEnv(env)

    env = ClipRewardEnv(env)

    env = WarpFrame(env, width=config.resolution[1], height=config.resolution[0], grayscale=config.grayscale)

    if instance == 0:
        env = RecorderWrapper(env, fps=BASE_FPS_ATARI // config.frame_skip, save_dir=config.save_dir, label='preproc', record_every=config.record_every)

    if decorr_steps is not None:
        env = DecorrEnvWrapper(env, decorr_steps)

    return env

def create_retro_env(config, instance_seed, instance, decorr_steps):
    use_restricted_actions = retro.Actions.FILTERED
    if config.retro_action_patch == 'discrete':
        use_restricted_actions = retro.Actions.DISCRETE

    print(f'Retro env has init states: {retro_utils.get_init_states()[config.env_name[6:]]}, using state: {config.retro_state}')
    randomize_state_on_reset = False
    retro_state = retro.State.DEFAULT if (config.retro_state in ['default', 'randomized']) else config.retro_state
    if config.retro_state == 'randomized': randomize_state_on_reset = True

    env = retro.make(config.env_name[6:], state=retro_state, use_restricted_actions=use_restricted_actions)
    if randomize_state_on_reset:  # note: this might mess with any EpisodicLifeEnv-like wrappers!
        env = RandomizeStateOnReset(env, instance_seed)

    if config.retro_action_patch == 'single_buttons':
        env = Discretizer(env, [[x] for x in env.unwrapped.buttons if x not in ('SELECT', 'START')])
    elif isinstance(config.retro_action_patch, list):
        # e.g. [[], ['LEFT'], ['RIGHT'], ['A'], ['B'], ['DOWN'], ['UP'], ['X'], ['Y'],
        #       ['LEFT', 'X'], ['RIGHT', 'X'], ['UP', 'X'], ['DOWN', 'X']]
        # or for sonic: [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        env = Discretizer(env, config.retro_action_patch)
    if config.retro_action_patch == 'auto':
        if 'SonicTheHedgehog'.lower() in config.env_name.lower():
            scheme = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]



        env = Discretizer(env, config.retro_action_patch)


    if instance == 0:
        env = RecorderWrapper(env, fps=BASE_FPS_ATARI, save_dir=config.save_dir, label='emulator', record_every=config.record_every)

    env = TimeLimit(env, max_episode_steps=config.time_limit)
    if config.frame_skip > 1:
        env = StochasticFrameSkip(env, seed=instance_seed, n=config.frame_skip, stickprob=config.retro_stickyprob)
    env = RecordEpisodeStatistics(env)
    env = RetroEpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = WarpFrame(env, width=config.resolution[1], height=config.resolution[0], grayscale=config.grayscale)

    if instance == 0:
        env = RecorderWrapper(env, fps=BASE_FPS_ATARI // config.frame_skip, save_dir=config.save_dir, label='preproc', record_every=config.record_every)

    if decorr_steps is not None:
        env = DecorrEnvWrapper(env, decorr_steps)
    return env

def create_procgen_env(config, instance_seed, instance):
    procgen_args = {k[8:]: v for k, v in vars(config).items() if k.startswith('procgen_')}
    procgen_args['start_level'] += 300_000*instance
    env = gym.make(f'procgen:procgen-{config.env_name.lower()[8:]}-v0', **procgen_args)

    if instance == 0:
        env = RecorderWrapper(env, fps=BASE_FPS_PROCGEN, save_dir=config.save_dir, label='emulator', record_every=config.record_every)

    if config.frame_skip > 1:
        print('Frame skipping for procgen enabled!')
        env = SkipFrameEnv(env, config.frame_skip)

    env = RecordEpisodeStatistics(env)

    # Note: openai doesn't use reward clipping for procgen with ppo & rainbow (https://arxiv.org/pdf/1912.01588.pdf)
    # env = ClipRewardEnv(env)

    env = WarpFrame(env, width=config.resolution[1], height=config.resolution[0], grayscale=config.grayscale)

    if instance == 0:
        env = RecorderWrapper(env, fps=BASE_FPS_PROCGEN // config.frame_skip, save_dir=config.save_dir, label='preproc', record_every=config.record_every)
    return env

def create_env_instance(args, instance, decorr_steps):
    instance_seed = args.seed+instance

    decorr_steps = None if decorr_steps is None else decorr_steps*instance

    if args.env_name.startswith('retro:'): env = create_retro_env(args, instance_seed, instance, decorr_steps)
    elif args.env_name.startswith('gym:'): env = create_atari_env(args, instance_seed, instance, decorr_steps)
    elif args.env_name.startswith('procgen:'): env = create_procgen_env(args, instance_seed, instance)

    if not args.env_name.startswith('procgen:'):
        env.seed(instance_seed)
        env.action_space.seed(instance_seed)
        env.observation_space.seed(instance_seed)
    return env

def create_env(args, decorr_steps=None):
    env_fns = [partial(create_env_instance, args=args, instance=i, decorr_steps=decorr_steps) for i in range(args.parallel_envs)]
    vec_env = partial(SubprocVecEnvNoFlatten) if args.subproc_vecenv else DummyVecEnvNoFlatten
    env = vec_env(env_fns)
    env = LazyVecFrameStack(env, args.frame_stack, args.parallel_envs, clone_arrays=not args.subproc_vecenv, lz4_compress=False)
    return env
