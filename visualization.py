import os
import argparse
import gym
import gym_simple_minigrid  # just to register envs
import numpy as np
import matplotlib.pyplot as plt

from lrgp.hierarchy import Hierarchy

# parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='Simple-MiniGrid-FourRooms-15x15-v0', help='Environment to use')
#
# args = parser.parse_args()
# args = vars(args)
#
# env = gym.make(args['env'])
#
# tester = Hierarchy(env)
# tester.load(os.path.join('checkpoints', args['checkpoint_name']))

logs = np.load('checkpoints/new_buffer_inner_loop_init/logs.npy')

print(len(logs))

episode = logs[:, 0]
subg = logs[:, 1]
subg_a = logs[:, 2]
steps = logs[:, 3]
steps_a = logs[:, 4]
max_subg = logs[:, 5]
sr = logs[:, 6]
low_sr = logs[:, 7]
bad_propose = logs[:, 8]
len_high_buffer = logs[:, 9]
len_low_buffer = logs[:, 10]
len_low_reachable_buffer = logs[:, 11]
len_low_allowed = logs[:, 12]

# len_high_buffer = logs[:, 8]
# len_low_buffer = logs[:, 9]
# len_low_reachable_buffer = logs[:, 10]
# len_low_allowed = logs[:, 11]

plt.plot(sr)
plt.xlabel('episode')
plt.ylabel('success rate high policy')

plt.plot(bad_propose)
plt.xlabel('episode')
plt.ylabel('success rate high policy')

plt.plot(len_low_buffer)
plt.xlabel('episode')
plt.ylabel('length buffer low policy')