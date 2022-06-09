#!/usr/bin/env python3
from importlib import import_module
import rospy
import math
import tf
import numpy
import roslib
import random
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Int8
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

# IMPORTS
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.nn as nn
import numpy as np
import albumentations as A
import torch.optim as optim
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
from PIL import Image as im
import torch.nn.functional as F
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

device = "cuda" if torch.cuda.is_available() else "cpu"

img=Image()
bridge=CvBridge()
rgbimg=Image()
depthimg=Image()
def depthcall(data):
    global img,f 
    try:
        img = bridge.imgmsg_to_cv2(data, "32FC1")
        depthimg=img
        cv_image_array = np.array(img, dtype = np.dtype('f8'))
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        #img has the depth values and cv_image_norm is img normalised from 0 to 1
        
    except CvBridgeError as e:
        print(e)

def rgbcall(data):
        #this is the rgb image u can use opencv functions on it
        img = bridge.imgmsg_to_cv2(data)
        rgbimg=img


# UNET

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

val_transforms = A.Compose(
    [
        A.Resize(height=128, width= 128),
        A.Normalize(
            mean = 0.0,
            std = 1.0, 
            max_pixel_value = 255.0,
        ),
        ToTensorV2(),
    ]
)

model = UNET(in_channels = 4, out_channels = 1).to(device)

model.load_state_dict(torch.load("10Kepochsunet114imagestrainedweights.pt"))

i = Image.fromarray(img)
i = i.resize((128, 128))
i = np.asarray(i)
if(i.shape[0]==3 or i.shape[1]==3 or i.shape[2]==3):
    li = [eh for eh in i]
    li.append(np.ones(128, 128))
    i = np.stack(li)

augmentations = val_transforms(image=i)
i = augmentations["image"]


i = i.unsqueeze(0).to(device)
ipred = model(i)

d = Image.fromarray(depthimg)
d = d.resize((128, 128))
d = np.asarray(d)
if(d.shape[0]==3 or d.shape[1]==3 or d.shape[2]==3):
    li = [eh for eh in d]
    li.append(np.ones(128, 128))
    i = np.stack(d)

daugmentations = val_transforms(image=d)
d = daugmentations["image"]


d = d.unsqueeze(0).to(device)
dpred = model(d)

bitand = np.bitwise_and(ipred, dpred)


rospy.init_node('depthdata', anonymous=True)
m1=rospy.Subscriber('/depth_camera/depth/image_raw',Image,depthcall,queue_size=1)
m2=rospy.Subscriber('/depth_camera/rgb/image_raw',Image,rgbcall,queue_size=1)
mask=rospy.Publisher('/inter_img',Image,queue_size=10)
rate=rospy.Rate(100)
while not rospy.is_shutdown():
        rate.sleep()