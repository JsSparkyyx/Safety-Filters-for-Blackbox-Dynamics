import numpy as np
import os
path = "./trajectories"
img = []
u = []
p = []
c = []
for i in range(1,5):
    data_path = os.path.join(path,str(i))
    imgs = np.load(os.path.join(data_path,'state.npy'))
    us = np.loadtxt(os.path.join(data_path,'velocity.txt'))
    collision = np.loadtxt(os.path.join(data_path,'collision.txt'))
    position = np.loadtxt(os.path.join(data_path,'position.txt'))
    u.append(us)
    img.append(imgs)
    c.append(collision)
    p.append(position)
    print(imgs.shape)
path = './trajectories/0'
np.save(os.path.join(path,'state.npy'),np.concatenate(img,axis=0))
np.savetxt(os.path.join(path,'velocity.txt'),np.concatenate(u,axis=0))
np.savetxt(os.path.join(path,'position.txt'),np.concatenate(p,axis=0))
np.savetxt(os.path.join(path,'collision.txt'),np.concatenate(c,axis=0))

