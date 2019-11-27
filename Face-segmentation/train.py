
import torch

from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
#from unet import Unet
from mydataset import MyDataset
from torch.utils import data

from model import Unet

image_path="/dataset/images/"
label_path="/dataset/labels/"
train_data="/dataset/train.txt"


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            print("outputs:",outputs.shape)
            print("labels:",labels.shape)
            break
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        break
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    return 
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

#训练模型
model = Unet(3, 1).to(device)
batch_size = 1
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())



dataset = MyDataset(image_path,label_path,train_data,transform=x_transforms, target_transform=y_transforms)
data_loader = data.DataLoader(dataset,  batch_size=1, shuffle=True, num_workers=0)

train_model(model, criterion, optimizer, data_loader)




