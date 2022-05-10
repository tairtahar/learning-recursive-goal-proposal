import numpy as np
import os
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_name', type=str, default='lrgp_four_rooms', help='Name of the checkpoint '
                                                                                   'subdirectory')

args = parser.parse_args()
args = vars(args)

data = np.load(os.path.join('logs', args['checkpoint_name'], 'logs.npy'))
x = np.linspace(0, np.shape(data)[0], np.shape(data)[0]) * 50

plt.plot(x,data[:,6])
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('Success ratio along training')
plt.show()