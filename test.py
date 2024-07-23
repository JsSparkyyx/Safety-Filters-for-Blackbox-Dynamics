from FastLoader import DeepAccidentDataset
from method.ViT import InDCBFTrainer, InDCBFController, Barrier
import torch
import time
import pytorch_lightning as pl
data = DeepAccidentDataset()
data.setup()
test_dataloader = data.val_dataloader()
barrier = Barrier(512,2)
model = InDCBFController(2,"cuda",model="google/vit-base-patch16-224",latent_dim=512)
checkpoint = torch.load("/root/tf-logs/DynamicLearning/version_1/checkpoints/last.ckpt")
trainer = InDCBFTrainer(model,barrier)
trainer.load_state_dict(checkpoint['state_dict'])
for idx, (i,u,label) in enumerate(test_dataloader):
    break
x_init = torch.zeros(1,512).to(i.device)
x_p = model.encoder(i[0,0,:].unsqueeze(0),x_init,u[0,0].unsqueeze(0))
print(model(i[0,1].unsqueeze(0),u[0,0].unsqueeze(0),x_p,u[0,0].unsqueeze(0),barrier))
print(u[0,0])
print(u[0,1])
# import cvxpy as cp
# import numpy as np
# u = cp.Variable(2)
# d_b = np.random.rand(512)
# f = np.random.rand(512)
# g = np.random.rand(512,2)
# t1 = d_b @ f
# print(t1)
# t2 = d_b @ g 
# print(t2)
# t3 = 1
# objective = cp.Minimize(cp.sum_squares(u - np.array([0,0])))
# constraints = [(t1+t2@u+t3)>=0]
# prob = cp.Problem(objective, constraints)
# result = prob.solve()
# print(result)

# from torchvision import datasets, transforms
# from torchvision.io import read_image
# import os
# import numpy as np
# from PIL import Image
# import torch 
# import time

# DATA_PATH = "/root/autodl-tmp/data/DeepAccident_data"
# CAMERA_LOC = ["Camera_Back","Camera_BackLeft","Camera_BackRight","Camera_Front","Camera_FrontLeft","Camera_FrontRight"]
# trajectory_list = np.loadtxt(os.path.join(DATA_PATH,'train.txt'),dtype=str)
# trajectory = trajectory_list[0]
# label_path = os.path.join(DATA_PATH,trajectory[0],'ego_vehicle/label',trajectory[1])
# label = 0
# transform = transforms.Compose([
#             transforms.Resize((224, 224))       
#         ])
# if "accident" in trajectory[0]:
#     label = 1
# imgs_all = []
# u = []
# t1=time.time()
# for i in range(1,101):
#     info_name = "{}_{:03d}.txt".format(trajectory[1],i)
#     info_path = os.path.join(label_path,info_name)
#     with open(info_path,'r') as f:
#         action = f.readline().strip().split(" ")
#     u.append(torch.Tensor([float(action[0]),float(action[1])]))
#     imgs = []
#     for cam_idx in CAMERA_LOC:
#         camera_path = os.path.join(DATA_PATH,trajectory[0],'ego_vehicle',cam_idx,trajectory[1])
#         img_name = "{}_{:03d}.jpg".format(trajectory[1],i)
#         img_path = os.path.join(camera_path,img_name)
#         img = read_image(img_path)/255
#         sample = transform(img)
#         imgs.append(sample)
#     imgs = torch.stack(imgs,dim=0)
#     imgs_all.append(imgs)
# result = torch.stack(imgs_all,dim=0)
# t2=time.time()
# print(t2-t1)
# trajectory = trajectory_list[0]
# camera_path = os.path.join(DATA_PATH,trajectory[0],'ego_vehicle/Camera_Front',trajectory[1])
# label_path = os.path.join(DATA_PATH,trajectory[0],'ego_vehicle/label',trajectory[1])
# transforms = transforms.Compose([
#             transforms.Resize((255, 255)),   
#             transforms.ToTensor(),              
#         ])

# label = 0
# if "accident" in trajectory[0]:
#     label = 1
# imgs = []
# u = []
# for i in range(1,101):
#     img_name = "{}_{:03d}.jpg".format(trajectory[1],i)
#     info_name = "{}_{:03d}.txt".format(trajectory[1],i)
#     img_path = os.path.join(camera_path,img_name)
#     info_path = os.path.join(label_path,info_name)
#     img = Image.open(img_path).convert('RGB')
#     with open(info_path,'r') as f:
#         action = f.readline().strip().split(" ")
#     u.append(torch.Tensor([float(action[0]),float(action[1])]))
#     sample = transforms(img)
#     imgs.append(sample)
# x = torch.stack(imgs,dim=0)
# u = torch.stack(u,dim=0)
# label = torch.LongTensor([label])
# print(x.shape)
# print(u.shape)
# print(label.shape)
# print(label)