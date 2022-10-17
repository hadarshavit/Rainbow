from common.env_wrappers import create_env_instance
from argparse import Namespace
import argparse
import numpy as np
from tqdm import tqdm

envs = ['Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=int)
    env_name = envs[parser.parse_args().env_id]
    print(env_name)
    for i in tqdm(range(50_000)):
        args = Namespace(env_name=f'gym:{env_name}', seed=1, time_limit=100_000, frame_skip=4, gamma=0.99, resolution=(224, 224), grayscale=True)
        env = create_env_instance(args, instance=i + 1, decorr_steps=0)
        done = False
        obs = env.reset()
        ram = env.unwrapped._get_ram()

        all_obserbatios = [obs]
        all_rams = []
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            ram = env.unwrapped._get_ram()
            all_obserbatios.append(state)
    np.save(f'./data1/s3092593/atari_rams/{env}.npy', np.array(all_obserbatios))