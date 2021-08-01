import random
from functools import partial
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from torch import nn as nn
from rich import print
from torch.cuda.amp import GradScaler, autocast

from common import networks
from common.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from common.utils import prep_observation_for_qnet

class Rainbow:
    buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env, args: SimpleNamespace) -> None:
        self.env = env
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp

        net = networks.get_model(args.network_arch)
        linear_layer = partial(networks.FactorizedNoisyLinear, sigma_0=args.noisy_sigma0) if args.noisy_dqn else nn.Linear
        depth = args.frame_stack*(1 if args.grayscale else 3)
        self.q_policy = net(depth, env.action_space.n, linear_layer, resolution=args.resolution).cuda()
        self.q_target = net(depth, env.action_space.n, linear_layer, resolution=args.resolution).cuda()
        self.q_target.load_state_dict(self.q_policy.state_dict())

        self.double_dqn = args.double_dqn

        self.prioritized_er = args.prioritized_er
        if self.prioritized_er:
            self.buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)
        else:
            self.buffer = UniformReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        self.n_step_gamma = args.gamma ** args.n_step

        self.max_grad_norm = args.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.decay_lr = args.lr_decay_steps is not None
        if self.decay_lr: self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)

        loss_fn_cls = nn.MSELoss if args.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none' if self.prioritized_er else 'mean'))

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    @torch.no_grad()
    def reset_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def act(self, states, eps: float):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                states = prep_observation_for_qnet(torch.from_numpy(np.stack(states)), self.use_amp)
                action_values = self.q_policy(states, advantages_only=True)
                actions = torch.argmax(action_values, dim=1)
            if eps > 0:
                for i in range(actions.shape[0]):
                    if random.random() < eps:
                        actions[i] = self.env.action_space.sample()
            return actions.cpu()

    @torch.no_grad()
    def td_target(self, reward: float, next_state, done: bool):
        self.reset_noise(self.q_target)
        if self.double_dqn:
            best_action = torch.argmax(self.q_policy(next_state, advantages_only=True), dim=1)
            next_Q = torch.gather(self.q_target(next_state), dim=1, index=best_action.unsqueeze(1)).squeeze()
            return reward + self.n_step_gamma * next_Q * (1 - done)
        else:
            max_q = torch.max(self.q_target(next_state), dim=1)[0]
            return reward + self.n_step_gamma * max_q * (1 - done)

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = self.buffer.sample(batch_size, beta)
            weights = torch.from_numpy(weights).cuda()
        else:
            state, next_state, action, reward, done = self.buffer.sample(batch_size)

        self.opt.zero_grad()
        with autocast(enabled=self.use_amp):
            td_est = torch.gather(self.q_policy(state), dim=1, index=action.unsqueeze(1)).squeeze()
            td_tgt = self.td_target(reward, next_state, done)

            if self.prioritized_er:
                td_errors = td_est-td_tgt
                new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
                self.buffer.update_priorities(indices, new_priorities)

                losses = self.loss_fn(td_tgt, td_est)
                loss = torch.mean(weights * losses)
            else:
                loss = self.loss_fn(td_tgt, td_est)

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.opt)
        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

        if self.decay_lr:
            self.scheduler.step()

        return td_est.mean().item(), loss.item(), grad_norm.item()

    def save(self, game_frame, **kwargs):
        save_path = (self.save_dir + f"/checkpoint_{game_frame}.pt")
        torch.save({**kwargs, 'state_dict': self.q_policy.state_dict(), 'game_frame': game_frame}, save_path)

        try:
            artifact = wandb.Artifact('saved_model', type='model')
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)
            print(f'Saved model checkpoint at {game_frame} frames.')
        except Exception as e:
            print('[bold red] Error while saving artifacts to wandb:', e)
