'''
Description: 
Date: 2022-11-24 14:41:03
LastEditTime: 2023-08-11 12:02:20
FilePath: /chengdongzhou/federatedLearning/SimpleProtoHAR/FLAlgorithms/trainmodel/cnnbase_fc.py
'''
import torch.nn as nn
from torch.nn import functional as F



class HARCNN(nn.Module):
    def __init__(self, data_name , conv_kernel_size = ( 9 , 1 ) ):
        super().__init__()
        channel_list = {
                        'uschad': [16, 32, 64, 7680, 20, 12 ],
                        'harbox': [16, 32, 64, 2176,  3,  5  ]
                        }
        down_sample = {
                        'ucihar': (2,2),
                        'pamap2': (2,2),
                        'wisdm' : (2,1),
                        'unimib': (2,1),
                        'hhar'  : (2,2),
                        'uschad': (2,2),
                        'harbox': (2,2),
                        }
        channel = channel_list[data_name]
        down    = down_sample[data_name]
        self.stem = nn.Sequential(
            nn.Conv2d(1, channel[0], kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = down )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = down )
        )
        self.flatten    = nn.Flatten()
        self.embedding  = nn.Linear( channel[3] , int(channel[3]//channel[4]) )
        self.classifier = nn.Linear( int(channel[3]//channel[4]) , channel[-1] )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        proto = F.relu( self.embedding( x ) )
        x = self.classifier(proto)

        return x,proto