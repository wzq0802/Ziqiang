from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
from net import *
import torch
import matplotlib.pyplot as plt



data_path = os.listdir("test")
data_path.sort(key=lambda x:int(x[:-4]))

def default_loader(path):
    data_pil =  np.load("test/%s"%(path)).reshape((1,256,256))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor)
    return data_tensor

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = data_path
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        return img
    def __len__(self):
        return len(self.images)


G = DualUNet().cuda()
mod = torch.load('DU/net.pth')
G.load_state_dict(mod)
G.eval()


train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=1)
i = 0
res = np.zeros((200,256,256))
for data in trainloader:
    with torch.no_grad():

        data = data.cuda()
        pred = G(data)

        pred = pred.cpu().numpy()
        print(pred)
        pred = np.where(pred<0.5,np.zeros_like(pred),np.ones_like(pred))

        res[i]=pred
        plt.imsave("DU/pred/%d.png"%(i),res[i].reshape(256,256),cmap="gray")
        i +=1

print(res.shape)
res.astype(np.uint8).tofile("DU/pred.raw")


