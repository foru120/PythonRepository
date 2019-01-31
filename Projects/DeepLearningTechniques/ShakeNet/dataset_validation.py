import os
import numpy as np
import PIL.Image as pilimg

DATA_ROOT_PATH = 'G:/04_dataset/cifar10_tt'

x = []

for (path, dirs, files) in os.walk(os.path.join(DATA_ROOT_PATH, 'test')):
    for file in files:
        x.append(np.asarray(pilimg.open(os.path.join(path, file))))

x_len = len(x)

x = np.asarray(x)

x = x/255.

print('>> Dataset mean: ', np.mean(x[:,:,:,0]), np.mean(x[:,:,:,1]), np.mean(x[:,:,:,2]))
print('>> Dataset std: ', np.std(x[:,:,:,0]), np.std(x[:,:,:,1]), np.std(x[:,:,:,2]))