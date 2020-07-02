import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import cv2

import torch.nn.functional as F

transform1 = transforms.Compose([
    transforms.Resize(60),
    transforms.CenterCrop(60),
    transforms.ToTensor()
])

class_names = ['#s001#', '#s002#', '#s003#', '#s004#', '#s005#', '#s006#', '#s007#', '#s008#', '#s009#', '#s010#',
               '#s011#', '#s012#', '#s013#', '#s014#', '#s015#', '#s016#', '#s017#', '#s018#', '#s019#', '#s020#',
               '#s021#', '#s022#', '#s023#', '#s024#', '#s025#', '#s026#', '#s027#', '#s028#', '#s029#', '#s030#',
               '#s031#', '#s032#', '#s033#', '#s034#', '#s035#', '#s036#', '#s037#', '#s038#', '#s039#', 'bg', '⊙', '①',
               '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '■', '□', '▲', '△', '◆', '◇', '◎', '●', '★', '☆', '一、']

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 62)
# model = model.to(device)
model = torch.nn.DataParallel(model).to(device)

PATH = './cifar_net29.pth'
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
lst_pics = []
path = "test_pic"
# path = "/data_880/zyc_data/CV/sign_image_gen/sign_data/val/#s001#"
for home, dirs, files in os.walk(path):
    files.sort()
    for filename in files:
        fullname = os.path.join(home, filename)
        # img = Image.open(fullname)
        img = cv2.imread(fullname, 1)
        img = Image.fromarray(np.uint8(img))

        img1 = transform1(img)
        img1 = img1.reshape(1, img1.shape[0], img1.shape[1], img1.shape[2])
        lst_pics.append(img1)

inputs = torch.cat(lst_pics).to(device)

with torch.set_grad_enabled(False):
    outputs0 = model(inputs)
    scores0, preds0 = torch.max(outputs0, 1)

    outputs = F.softmax(outputs0, dim=1)
    scores, preds = torch.max(outputs, 1)
    print(preds)
    a = outputs.numpy()  # np.max(a,axis=1)>9
    num = 0
    for home, dirs, files in os.walk(path):
        files.sort()
        for filename in files:
            fullname = os.path.join(home, filename)
            print("{}\t{}\t{}\t{}".format(filename, class_names[preds[num]],
                                          scores0[num].numpy(),
                                          scores[num].numpy()))
            num += 1
