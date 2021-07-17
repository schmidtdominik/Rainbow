import os
import random
import socket
import sys
import zlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as sn

import torch
from tqdm.auto import trange

import common.retro_utils
from common import retro_utils
from common.env_wrappers import create_env


def prep_observation_for_qnet(tensor, use_amp):
    """ Tranfer the tensor the gpu and reshape it into (batch, frame_stack*channels, y, x) """
    assert len(tensor.shape) == 5, tensor.shape # (batch, frame_stack, y, x, channels)
    tensor = tensor.cuda().permute(0, 1, 4, 2, 3) # (batch, frame_stack, channels, y, x)
    # .cuda() needs to be before this ^ so that the tensor is made contiguous on the gpu
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1]*tensor.shape[2], *tensor.shape[3:]))

    return tensor.to(dtype=(torch.float16 if use_amp else torch.float32)) / 255

class LinearSchedule:

    def __init__(self, burnin: int, initial_value: float, final_value: float, decay_time: int):
        """ Linearly decaying function """
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_time = decay_time
        self.burnin = burnin

    def __call__(self, frame: int) -> float:
        if frame < self.burnin:
            return self.initial_value
        else:
            frame = frame - self.burnin

        slope = (self.final_value - self.initial_value) / self.decay_time
        if self.final_value < self.initial_value:
            return max(slope * frame + self.initial_value, self.final_value)
        else:
            return min(slope * frame + self.initial_value, self.final_value)

def get_mean_ep_length(args):
    dc_args = deepcopy(args)
    dc_args.parallel_envs = 12
    dc_args.subproc_vecenv = True
    dc_env = create_env(dc_args)
    dc_env.reset()

    # Decorrelate envs
    ep_lengths = []
    for frame in trange(args.time_limit//4+100):
        _, _, _, infos = dc_env.step([dc_env.action_space.sample() for x in range(dc_args.parallel_envs)])
        for info, j in zip(infos, range(dc_args.parallel_envs)):
            if 'episode_metrics' in info.keys():
                ep_lengths.append(info['episode_metrics']['length'])

    dc_env.close()
    mean_length = sum(ep_lengths)/len(ep_lengths)
    print(f'Mean episode length for this env is {mean_length} (computed over {len(ep_lengths)} episodes).')
    return mean_length

def get_seed(o: str, i):
    return i + zlib.adler32(bytes(o, encoding='utf-8')) % 10000

def prep_args(args):
    # otherwise target may not be synced since the main loop iterates in steps of parallel_envs
    assert (args.sync_dqn_target_every % args.parallel_envs) == 0
    assert args.loss_fn in ('mse', 'huber')
    assert (args.lr_decay_steps is None) == (args.lr_decay_factor is None)
    assert args.burnin > args.batch_size
    assert (args.retro_dense_id is None or args.random_retro_env is None)

    if args.random_retro_env is not None:
        if args.random_retro_env == 'random':
            args.env_name = 'retro:' + random.choice(retro_utils.dense_envs)
        else:
            assert args.random_retro_env == 'random-multistate'
            init_states = retro_utils.get_init_states()
            args.env_name = 'retro:' + random.choice([x for x in retro_utils.dense_envs if len(init_states[x]) > 1])

        args.retro_state = 'randomized'
        print(f'Selected {args.env_name} as random retro env.')
    elif args.retro_dense_id is not None:
        args.env_name = 'retro:' + retro_utils.dense_envs[args.retro_dense_id]

    if args.adam_eps is None:
        args.adam_eps = 0.005/args.batch_size

    if args.prioritized_er_time is None:
        args.prioritized_er_time = args.training_frames

    user_seed = args.seed
    args.seed = get_seed(args.env_name, args.seed)

    if args.env_name.startswith('gym:'):
        if args.frame_skip is None: args.frame_skip = 4
        if args.frame_stack is None: args.frame_stack = 4
        args.resolution = (84, 84)
        if args.grayscale is None: args.grayscale = True

        if args.record_every is None: args.record_every = 60 * 50
    elif args.env_name.startswith('retro:'):
        if args.frame_skip is None: args.frame_skip = 4
        if args.frame_stack is None: args.frame_stack = 4
        args.resolution = (72, 88)
        if args.grayscale is None: args.grayscale = False

        if not args.subproc_vecenv: print('[WARNING] subproc_vecenv was forcibly enabled since this is a retro env!')
        args.subproc_vecenv = True

        if args.record_every is None: args.record_every = 60 * 50

    elif args.env_name.startswith('procgen:'):
        if args.frame_skip is None: args.frame_skip = 1
        if args.frame_stack is None: args.frame_stack = 4
        args.resolution = args.resolution = (64, 64)
        if args.grayscale is None: args.grayscale = False
        if args.record_every is None: args.record_every = 60 * 50

        args.time_limit = None

    # hyperparameters for DER are adapted from https://github.com/Kaixhin/Rainbow
    if args.der:
        args.parallel_envs = 1
        args.batch_size = 32
        args.train_count = 1

        args.burnin = 6400
        args.n_step = 20
        args.sync_dqn_target_every = 8000
        args.buffer_size = 2 ** 17  # or 2**16
        args.lr = 0.00025
        args.adam_eps = 0.0003125

    if args.noisy_dqn:
        args.init_eps = 0.0
        args.final_eps = 0.0
        #args.eps_decay_frames = 20_000

    # check where we are executing
    in_colab = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ or os.path.exists('/colabtools')
    args.instance = 'colab' if in_colab else socket.gethostname()

    wandb_log_config = deepcopy(vars(args))
    wandb_log_config['env_type'] = args.env_name[:args.env_name.find(':')]
    wandb_log_config['seed'] = user_seed
    del wandb_log_config['training_frames']
    del wandb_log_config['record_every']
    del wandb_log_config['use_wandb']
    del wandb_log_config['run_desc']

    if not args.env_name.startswith('retro:'):
        for k in list(wandb_log_config.keys()):
            if k.startswith('retro'):
                del wandb_log_config[k]
    if not args.env_name.startswith('procgen:'):
        for k in list(wandb_log_config.keys()):
            if k.startswith('procgen'):
                del wandb_log_config[k]
    tags = ['dataeff'] if args.training_frames == 100_000 else []

    return args, tags, wandb_log_config
