# -*- coding: utf-8 -*-
"""

@author: Nicolás cubero Torres
@description: Utility for visualization and labeling of a video
    dataset directory
@usage: label.py <videos directory>
"""
# Modules imported
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models import istl

# Constants
CUBOIDS_LENGTH = 8

### Input arguments ###
if len(sys.argv) != 2:
    print('Usage: {} <videos directory>'.format(sys.argv[0]), file=sys.stderr)
    exit(1)

v_dir = sys.argv[1]
v_dir += '/' if not v_dir.endswith('/') else ''

### Data loading ###
data = istl.CuboidsGenerator(source=v_dir, cub_frames=CUBOIDS_LENGTH)
labels = np.zeros(len(data), dtype=np.int8)
exit(0)
for i in range(len(data)):

    # Play each cuboid
    vis = istl.FramesFromCuboidsGen(CUBOIDS_LENGTH, data[i].reshape(1,
                                                                *data[i].shape))
    fig, ax = plt.subplots()
    frames = [[ax.imshow(vis[j].astype('int32'),
                    animated=True)] for j in range(len(vis))]

    anim = animation.ArtistAnimation(fig, frames, repeat=True, interval=200,
                                        repeat_delay=100)
    plt.show()

    # Ask for the label of cuboid
    lab = None

    while lab is None or lab < 0 or lab > 1:
        try:
            lab = int(input('For cuboid {}, frames [{}-{}]. Type 1 for anomaly or 0 '\
                        'for normal: '.format(i, i*CUBOIDS_LENGTH,
                        i*CUBOIDS_LENGTH+CUBOIDS_LENGTH-1)))
        except:
            print('Please, type 1 for anomaly or 0 for normal')
            continue

        if lab < 0 or lab > 1:
            print('Please, type 1 for anomaly or 0 for normal')

    labels[i] = lab

np.savetxt(v_dir+'labels.txt', labels)
