import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ColorImage(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(50):
            filenames = glob.glob(os.path.join(root, str(i)+'_'+'*.png'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        vgg16 = models.vgg16_bn(pretrained=True)
        vgg16.classifier[6] = nn.Linear(4096, 50)
        vgg16.classifier[6].requires_grad = True
        self.vgg = vgg16
        
    def forward(self, input):
        x = self.vgg(input)
        return x

def save_checkpoint(checkpoint_path, model):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


def main():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    train_set = ColorImage(root='hw2-ben980828/hw2_data/p1_data/train_50',
        transform=transforms.Compose([ 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ]))

    validation_set = ColorImage(root='hw2-ben980828/hw2_data/p1_data/val_50',
        transform=transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ]))

    # print('# in train set:', len(train_set))
    # print('# in validate set:', len(validation_set))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False, num_workers=1)

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    # print('Image tensor in each batch:', images.shape, images.dtype)
    # print('Label tensor in each batch:', labels.shape, labels.dtype)

    model = Net()
    model.to(device)
    # state = torch.load('p1model-4800.pth')
    # model.load_state_dict(state['state_dict'])

    
    epoch = 40
    log_interval = 100
    save_interval = 800
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    iteration = 0

    for ep in range(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('p1model-%i.pth' % iteration, model)
            iteration += 1
#################################### Evaluation #########################################
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target)
                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()
                    
        val_loss /= len(validation_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(validation_loader.dataset),
            100. * correct / len(validation_loader.dataset)))
    save_checkpoint('p1model-%i.pth' % iteration, model)



if __name__ == '__main__':
    main()
