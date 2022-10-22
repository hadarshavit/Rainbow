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
    wandb.init(project='rainbow_pre_train', save_code=True, config=dict(**wandb_log_config, log_version=100),
               mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=[args.wandb_tag] if args.wandb_tag else [])
    save_dir = Path(args.save_dir) / wandb.run.name
    save_dir.mkdir(parents=True)
    args.save_dir = str(save_dir)

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
    discounted_returns = deque(maxlen=10)
    losses = deque(maxlen=10)
    q_values = deque(maxlen=10)
    grad_norms = deque(maxlen=10)
    iter_times = deque(maxlen=10)
    reward_density = 0

    returns_all = []
    q_values_all = []

    # main training loop:
    # we will do a total of args.training_frames/args.parallel_envs iterations
    # in each iteration we perform one interaction step in each of the args.parallel_envs environments,
    # and args.train_count training steps on batches of size args.batch_size
    t = trange(0, args.training_frames + 1, args.parallel_envs)
    for game_frame in t:
        iter_start = time.time()
    
        actions = rainbow.act(states, eps)
        env.step_async(actions)

        # if any of the envs finished an episode, log stats to wandb
        for info, j in zip(infos, range(args.parallel_envs)):
            if 'episode_metrics' in info.keys():
                episode_metrics = info['episode_metrics']
                returns.append(episode_metrics['return'])
                returns_all.append((game_frame, episode_metrics['return']))
                discounted_returns.append(episode_metrics['discounted_return'])

                log = {'x/game_frame': game_frame + j, 'x/episode': episode_count,
                       'ep/return': episode_metrics['return'], 'ep/length': episode_metrics['length'], 'ep/time': episode_metrics['time'],
                       'ep/mean_reward_per_frame': episode_metrics['return'] / (episode_metrics['length'] + 1), 'grad_norm': np.mean(grad_norms),
                       'mean_loss': np.mean(losses), 'mean_q_value': np.mean(q_values), 'fps': args.parallel_envs / np.mean(iter_times),
                       'running_avg_return': np.mean(returns), 'lr': rainbow.opt.param_groups[0]['lr'], 'reward_density': reward_density,
                       'discounted_return': np.mean(discounted_returns)}
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
        if game_frame % (500_000-(500_000 % args.parallel_envs)) == 0 and game_frame > 0:
            rainbow.save(game_frame, args=args, run_name=wandb.run.name, run_id=wandb.run.id, target_metric=np.mean(returns), returns_all=returns_all, q_values_all=q_values_all)
            print(f'Model saved at {game_frame} frames.')

        iter_times.append(time.time() - iter_start)
        t.set_description(f' [{game_frame:>8} frames, {episode_count:>5} episodes]', refresh=False)

    wandb.log({'x/game_frame': game_frame + args.parallel_envs, 'x/episode': episode_count,
               'x/train_step': (game_frame + args.parallel_envs) // args.parallel_envs * args.train_count,
               'x/emulator_frame': (game_frame + args.parallel_envs) * args.frame_skip})
    env.close()
    wandb.finish()
