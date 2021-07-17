import random
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn

import torch, wandb
import numpy as np
from tqdm.auto import trange
from rich import print

from common import argp
from common.rainbow import Rainbow
from common.env_wrappers import create_env, BASE_FPS_ATARI, BASE_FPS_PROCGEN
from common.utils import LinearSchedule, prep_args, get_mean_ep_length

torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

"""

    Rainbow DQN
    - Implementation of the "Rainbow DQN" algorithm presented in Hessel, et al. (2017).
    - Includes all components apart from distributional RL (we saw mixed results with C51 and QR-DQN).
    - The training framework supports >1000 environments from gym, gym-retro and procgen.
    - By default the large IMPALA CNN with 2x channels from Espeholt et al. (2018) is used.
    - To reduce training time, the implementation uses large, vectorized environments and larger batch sizes.
    
    When trained for 10M frames, this implementation outperforms
        * google/dopamine           (trained for  10M frames)       on 96% of games
        * google/dopamine           (trained for 200M frames)       on 64% of games
        * Hessel, et al. (2017)     (trained for 200M frames)       on 40% of games
        * Human results                                             on 72% of games
    
    Most of the performance improvements compared to the paper come from the IMPALA CNN as well as some 
    hyperparameter changes (e.g. 4x larger learning rate).
    Full results available here: https://docs.google.com/spreadsheets/d/1ncCFIno4o83JmosAwj30XvIfWSIbO5btomfTrzEr4xE
    
    Some notes & tips
    - With a single RTX 3090 and 12 CPU cores, training for 10M frames takes around 8-10 hours, depending on the used settings
    - About 15GB RAM are required. When using a larger replay buffer, subprocess envs or larger resolutions, memory use may be *much* higher
    - hyperparameters can be configured through command line arguments, defaults are saved in `common/argp.py`
    - for fastest training throughput use batch_size=512, parallel_envs=64, train_count=1, subproc_vecenv=True
    
"""

if __name__ == '__main__':
    args, tags, wandb_log_config = argp.read_args()

    # set up logging
    wandb.init(project='konsaka', save_code=True, config={'log_version': 16, **wandb_log_config},
               tags=tags, job_type='train', notes=args.run_desc, mode=('online' if args.use_wandb else 'offline'))
    save_dir = Path("checkpoints") / wandb.run.name
    save_dir.mkdir(parents=True)
    args.save_dir = str(save_dir)

    # create decay schedules for dqn's exploration epsilon, per's beta parameter and the transition discard probability
    eps_schedule = LinearSchedule(0, initial_value=args.init_eps, final_value=args.final_eps, decay_time=args.eps_decay_frames)
    per_beta_schedule = LinearSchedule(0, initial_value=args.prioritized_er_beta0, final_value=1.0, decay_time=args.prioritized_er_time)
    tr_discard_prob = LinearSchedule(0, initial_value=args.tr_discard_prob, final_value=0.0, decay_time=args.tr_discard_time)

    # when using many (e.g. 64) environments in parallel, having all of them be correlated can be a problem
    # to avoid this problem we estimate the mean episode length for this game and then take i*(mean ep length/parallel envs count)
    # random steps in the i'th environment (this is done during the reset step further below)
    decorr_steps = None
    if args.decorr_envs and not args.env_name.startswith('procgen:'):
        decorr_steps = get_mean_ep_length(args)//args.parallel_envs

    print(f'Creating {args.parallel_envs} environment instances.. ', end='')
    env = create_env(args, decorr_steps=decorr_steps)
    print('Done.')

    print(f'Resetting and possibly decorrelating env (decorr_steps={decorr_steps}).. ', end='')
    states = env.reset()
    print('Done.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rainbow = Rainbow(env, args)
    wandb.watch(rainbow.q_policy)

    print('[blue bold]Running environment =', args.env_name,
          '[blue bold]\nwith action space   =', env.action_space,
          '[blue bold]\nobservation space   =', env.observation_space,
          '[blue bold]\nand config:', sn(**wandb_log_config))

    episode = 0
    returns = deque(maxlen=100)
    losses = deque(maxlen=20)
    q_values = deque(maxlen=20)
    iter_times = deque(maxlen=20)
    reward_density = 0

    for game_frame in trange(0, args.training_frames+1, args.parallel_envs):
        iter_start = time.time()
        eps = eps_schedule(game_frame)
        per_beta = per_beta_schedule(game_frame)

        # reset the noisy-nets noise in the policy
        if args.noisy_dqn:
            rainbow.reset_noise(rainbow.q_policy)

        # compute actions for all parallel envs
        actions = rainbow.act(states, eps)
        env.step_async(actions)

        # if training has started, perform args.train_count training steps, each on a batch of size args.batch_size
        if rainbow.buffer.burnedin:
            for train_iter in range(args.train_count):
                if args.noisy_dqn and train_iter > 0: rainbow.reset_noise(rainbow.q_policy)
                q, loss = rainbow.train(args.batch_size, beta=per_beta)
                losses.append(loss)
                q_values.append(q)

        # see https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/launcher.py#L155
        if game_frame % args.sync_dqn_target_every == 0 and rainbow.buffer.burnedin:
            rainbow.sync_Q_target()

        # block until environments are ready, then collect transitions and add them to the buffer
        next_states, rewards, dones, infos = env.step_wait()
        for state, action, reward, done, j in zip(states, actions, rewards, dones, range(args.parallel_envs)):
            reward_density = 0.999*reward_density + (1-0.999)*(reward != 0)
            if reward != 0 or random.random() > tr_discard_prob(game_frame):
                rainbow.buffer.put(state, action, reward, done, j=j)
        states = next_states

        # if any of the envs finished an episode, log that data to wandb
        for info, j in zip(infos, range(args.parallel_envs)):
            if 'episode_metrics' in info.keys():
                ep_return = info['episode_metrics']['return']
                ep_length = info['episode_metrics']['length']
                ep_time = info['episode_metrics']['time']
                returns.append(ep_return)

                log = {'x/game_frame': game_frame+j, 'x/episode': episode, 'x/train_step': (game_frame+j)//args.parallel_envs*args.train_count, 'x/emulator_frame': (game_frame+j) * args.frame_skip,
                       'ep/return': ep_return, 'ep/length': ep_length, 'ep/time': ep_time, 'ep/mean_reward_per_frame': ep_return/(ep_length+1),
                       'mean_loss': np.mean(losses), 'mean_q_value': np.mean(q_values), 'fps': args.parallel_envs/np.mean(iter_times),
                       'target_metric': np.mean(returns), 'lr': rainbow.opt.param_groups[0]['lr'], 'reward_density': reward_density}
                       #'actual_batch_size': actual_batch_size, 'actual_train_count': actual_train_count}
                if args.prioritized_er: log['per_beta'] = per_beta
                if eps > 0: log['epsilon'] = eps

                if 'emulator_recording' in info: log['emulator_recording'] = wandb.Video(info['emulator_recording'], fps=(BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI), format="mp4")
                if 'preproc_recording' in info: log['preproc_recording'] = wandb.Video(info['preproc_recording'], fps=(BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI)//args.frame_skip, format="mp4")

                wandb.log(log)
                episode += 1

        if game_frame % 400_000 == 0 and game_frame > 0:
            print(f'Target metric = {np.mean(returns)}')
            rainbow.save(game_frame, args=args, run_name=wandb.run.name, run_id=wandb.run.id, target_metric=np.mean(returns))
        iter_times.append(time.time()-iter_start)

    rainbow.save(game_frame+args.parallel_envs, args=args, run_name=wandb.run.name, run_id=wandb.run.id, target_metric=np.mean(returns))
    wandb.log({'x/game_frame': game_frame+args.parallel_envs, 'x/episode': episode, 'x/train_step': (game_frame+args.parallel_envs)//args.parallel_envs*args.train_count, 'x/emulator_frame': (game_frame+args.parallel_envs) * args.frame_skip})
    env.close()
    wandb.finish()
