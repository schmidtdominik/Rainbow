"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import time, random
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn

import torch, wandb
import numpy as np
from tqdm import trange
from rich import print

from common import argp
from common.rainbow import Rainbow
from common.env_wrappers import create_env, BASE_FPS_ATARI, BASE_FPS_PROCGEN
from common.utils import LinearSchedule, get_mean_ep_length

torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

if __name__ == '__main__':
    args, wandb_log_config = argp.read_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set up logging & model checkpoints
    wandb.init(project='rainbow', save_code=True, config=wandb_log_config,
               mode=('online' if args.use_wandb else 'offline'), anonymous='allow')
    save_dir = Path("checkpoints") / wandb.run.name
    save_dir.mkdir(parents=True)
    args.save_dir = str(save_dir)

    # create decay schedules for dqn's exploration epsilon and per's importance sampling (beta) parameter
    eps_schedule = LinearSchedule(0, initial_value=args.init_eps, final_value=args.final_eps, decay_time=args.eps_decay_frames)
    per_beta_schedule = LinearSchedule(0, initial_value=args.prioritized_er_beta0, final_value=1.0, decay_time=args.prioritized_er_time)

    # When using many (e.g. 64) environments in parallel, having all of them be correlated can be an issue.
    # To avoid this, we estimate the mean episode length for this environment and then take i*(mean ep length/parallel envs count)
    # random steps in the i'th environment.
    print(f'Creating', args.parallel_envs, 'and decorrelating environment instances. This may take up to a few minutes.. ', end='')
    decorr_steps = None
    if args.decorr and not args.env_name.startswith('procgen:'):
        decorr_steps = get_mean_ep_length(args) // args.parallel_envs
    env = create_env(args, decorr_steps=decorr_steps)
    states = env.reset()
    print('Done.')

    rainbow = Rainbow(env, args)
    wandb.watch(rainbow.q_policy)

    print('[blue bold]Running environment =', args.env_name,
          '[blue bold]\nwith action space   =', env.action_space,
          '[blue bold]\nobservation space   =', env.observation_space,
          '[blue bold]\nand config:', sn(**wandb_log_config))

    episode_count = 0
    returns = deque(maxlen=100)
    losses = deque(maxlen=10)
    q_values = deque(maxlen=10)
    grad_norms = deque(maxlen=10)
    iter_times = deque(maxlen=10)
    reward_density = 0

    # main training loop:
    # we will do a total of args.training_frames/args.parallel_envs iterations
    # in each iteration we perform one interaction step in each of the args.parallel_envs environments,
    # and args.train_count training steps on batches of size args.batch_size
    t = trange(0, args.training_frames + 1, args.parallel_envs)
    for game_frame in t:
        iter_start = time.time()
        eps = eps_schedule(game_frame)
        per_beta = per_beta_schedule(game_frame)

        # reset the noisy-nets noise in the policy
        if args.noisy_dqn:
            rainbow.reset_noise(rainbow.q_policy)

        # compute actions to take in all parallel envs, asynchronously start environment step
        actions = rainbow.act(states, eps)
        env.step_async(actions)

        # if training has started, perform args.train_count training steps, each on a batch of size args.batch_size
        if rainbow.buffer.burnedin:
            for train_iter in range(args.train_count):
                if args.noisy_dqn and train_iter > 0: rainbow.reset_noise(rainbow.q_policy)
                q, loss, grad_norm = rainbow.train(args.batch_size, beta=per_beta)
                losses.append(loss)
                grad_norms.append(grad_norm)
                q_values.append(q)

        # copy the Q-policy weights over to the Q-target net
        # (see also https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/launcher.py#L155)
        if game_frame % args.sync_dqn_target_every == 0 and rainbow.buffer.burnedin:
            rainbow.sync_Q_target()

        # block until environments are ready, then collect transitions and add them to the replay buffer
        next_states, rewards, dones, infos = env.step_wait()
        for state, action, reward, done, j in zip(states, actions, rewards, dones, range(args.parallel_envs)):
            reward_density = 0.999 * reward_density + (1 - 0.999) * (reward != 0)
            rainbow.buffer.put(state, action, reward, done, j=j)
        states = next_states

        # if any of the envs finished an episode, log stats to wandb
        for info, j in zip(infos, range(args.parallel_envs)):
            if 'episode_metrics' in info.keys():
                episode_metrics = info['episode_metrics']
                returns.append(episode_metrics['return'])

                log = {'x/game_frame': game_frame + j, 'x/episode': episode_count,
                       'ep/return': episode_metrics['return'], 'ep/length': episode_metrics['length'], 'ep/time': episode_metrics['time'],
                       'ep/mean_reward_per_frame': episode_metrics['return'] / (episode_metrics['length'] + 1), 'grad_norm': np.mean(grad_norms),
                       'mean_loss': np.mean(losses), 'mean_q_value': np.mean(q_values), 'fps': args.parallel_envs / np.mean(iter_times),
                       'running_avg_return': np.mean(returns), 'lr': rainbow.opt.param_groups[0]['lr'], 'reward_density': reward_density}
                if args.prioritized_er: log['per_beta'] = per_beta
                if eps > 0: log['epsilon'] = eps

                # log video recordings if available
                if 'emulator_recording' in info: log['emulator_recording'] = wandb.Video(info['emulator_recording'], fps=(
                    BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI), format="mp4")
                if 'preproc_recording' in info: log['preproc_recording'] = wandb.Video(info['preproc_recording'],
                    fps=(BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI) // args.frame_skip, format="mp4")

                wandb.log(log)
                episode_count += 1

        if game_frame % (50_000-(50_000 % args.parallel_envs)) == 0:
            print(f' [{game_frame:>8} frames, {episode_count:>5} episodes] running average return = {np.mean(returns)}')
            torch.cuda.empty_cache()

        # every 1M frames, save a model checkpoint to disk and wandb
        if game_frame % 1_000_000 == 0 and game_frame > 0:
            rainbow.save(game_frame, args=args, run_name=wandb.run.name, run_id=wandb.run.id, target_metric=np.mean(returns))

        iter_times.append(time.time() - iter_start)
        t.set_description(f' [{game_frame:>8} frames, {episode_count:>5} episodes]', refresh=False)

    rainbow.save(game_frame + args.parallel_envs, args=args, run_name=wandb.run.name, run_id=wandb.run.id, target_metric=np.mean(returns))
    wandb.log({'x/game_frame': game_frame + args.parallel_envs, 'x/episode': episode_count,
               'x/train_step': (game_frame + args.parallel_envs) // args.parallel_envs * args.train_count,
               'x/emulator_frame': (game_frame + args.parallel_envs) * args.frame_skip})
    env.close()
    wandb.finish()
