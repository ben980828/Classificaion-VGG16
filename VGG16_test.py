import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import sys
from PIL import Image
import torchvision.models as models
import numpy as np
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=50, bias=True)
        vgg16.classifier[6].requires_grad = True
        self.vgg = vgg16
        
    def forward(self, input):
        x = self.vgg(input)
        return x

pyfile = sys.argv[0]
input_folder = sys.argv[1]
output_folder = sys.argv[2]

# input_folder = 'hw2-ben980828/hw2_data/p1_data/val_50'
# output_folder = 'hw2-ben980828'

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

model = Net()

state = torch.load('p1model-4800_acc84.pth')
model.load_state_dict(state['state_dict'])
print(model)
model.to(device)
model.eval()
input_list = []
test_path = os.listdir(input_folder)
for fn in test_path:
    input_list.append(fn)
with open(os.path.join(output_folder, 'test_pred.csv'), 'w') as csvfile:
    csvfile.write('image_id,label\n')
    for i, filename in enumerate(input_list):# every image in folder(abs path)
        abs_path = os.path.join(input_folder, filename)
        pil_image = Image.open(abs_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
          ])
        image = transform(pil_image)
        image_ch = torch.unsqueeze(image, 0)
        model_input = image_ch.to(device)
        output = model(model_input)
        output_label = torch.argmax(output)
        label = output_label.item()
        csvfile.write('{},{}\n'.format(filename, label))