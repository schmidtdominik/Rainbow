from copy import deepcopy
import torch
from tqdm.auto import trange

from common.env_wrappers import create_env


def prep_observation_for_qnet(tensor, use_amp):
    """ Tranfer the tensor the gpu and reshape it into (batch, frame_stack*channels, y, x) """
    assert len(tensor.shape) == 5, tensor.shape # (batch, frame_stack, y, x, channels)
    tensor = tensor.cuda().permute(0, 1, 4, 2, 3) # (batch, frame_stack, channels, y, x)
    # .cuda() needs to be before this ^ so that the tensor is made contiguous on the gpu
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1]*tensor.shape[2], *tensor.shape[3:]))

    return tensor.to(dtype=(torch.float16 if use_amp else torch.float32)) / 255

class LinearSchedule:
    """Set up a linear hyperparameter schedule (e.g. for dqn's epsilon parameter)"""

    def __init__(self, burnin: int, initial_value: float, final_value: float, decay_time: int):
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
    """Run a few iterations of the environment and estimate the mean episode length"""
    dc_args = deepcopy(args)
    dc_args.parallel_envs = 6
    dc_args.subproc_vecenv = True
    dc_env = create_env(dc_args)
    dc_env.reset()

    # Decorrelate envs
    ep_lengths = []
    for frame in trange(args.time_limit//4+100):
        _, _, _, infos = dc_env.step([dc_env.action_space.sample() for x in range(dc_args.parallel_envs)])
        for info, j in zip(infos, range(dc_args.parallel_envs)):
            if 'episode_metrics' in info.keys(): ep_lengths.append(info['episode_metrics']['length'])
    dc_env.close()
    mean_length = sum(ep_lengths)/len(ep_lengths)
    return mean_length