import sys
import time
import warnings
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from net import *
import numpy as np
warnings.filterwarnings('ignore')
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


data_path = os.listdir("train")
data_path.sort(key=lambda x: int(x[:-4]))
data_path1 = os.listdir("label")
data_path1.sort(key=lambda x: int(x[:-4]))


train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]

def default_loader(path):
    data_pil = np.load("train/%s" % (path)).reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor)
    return data_tensor

def default_loader1(path):
    data_pil1 = np.load("label/%s" % (path)).reshape((1, 256, 256))
    data_tensor1 = torch.tensor(data_pil1).type(torch.FloatTensor)
    return data_tensor1


class trainset(Dataset):
    def __init__(self, paths, loader=default_loader, loader1=default_loader1):
        self.images = paths
        self.loader = loader
        self.loader1 = loader1


    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.loader1(fn)

        return img, target

    def __len__(self):
        return len(self.images)



train_data = trainset(train_paths)
val_data = trainset(val_paths)


batch_size = 64
# batch_size = 16

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)

# lambda_reg = 1e-5   ##Regularization to prevent overfitting on small datasets


net=DualUNet().cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
# optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999),weight_decay=1e-5)   ##Regularization to prevent overfitting on small datasets

mse = nn.MSELoss()
epochs = 200
Loss_list = []
Val_loss_list = []


for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    i = 0
    net.train()
    t1 = time.time()
    average_train_loss = 0.0
    for data, label in trainloader:
        i += 1
        data = data.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        pred = net(data)
        loss = mse(pred, label)

        # l2_reg = 0.0
        # for param in net.parameters():
        #     l2_reg += torch.norm(param, p=2)
        # loss += lambda_reg * l2_reg    ##Regularization to prevent overfitting on small datasets

        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())

        sys.stdout.write(
            "[Epoch %d/%d] [Batch:%d/%d] [loss: %f]\n" % (
                epoch, epochs, len(trainloader), i, loss.item(),
                ))

    Loss_list.append(train_loss / len(trainloader))

    net.eval()
    val_loss = 0
    with torch.no_grad():
        for data_val, label_val in valloader:
            data_val = data_val.cuda()
            label_val = label_val.cuda()
            pred_val = net(data_val)
            val_loss += float((mse(pred_val, label_val)).item())
    val_loss /= len(valloader)
    Val_loss_list.append(val_loss)

    t2 = time.time()
    print("Epoch time:", t2 - t1)


    if epoch % 100 == 0 and epoch != 0:
        torch.save(net.state_dict(), f"DU/net_{epoch}.pth")


torch.save(net.state_dict(), "DU/net.pth")


np.savetxt("DU/Train Loss.csv", np.array(Loss_list))
np.savetxt("DU/Val Loss.csv", np.array(Val_loss_list))


