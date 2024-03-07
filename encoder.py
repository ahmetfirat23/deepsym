import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from utils import gumbel_sigmoid

class SingleObjectEncoder(nn.Module):
    def __init__(self, interacting):
        super(SingleObjectEncoder, self).__init__()
        self.interacting = interacting
        self.input_size = 1
        self.output_size = 2
        self.conv3x3_1 = ConvolutionBlock([self.input_size,32], [32,32], 3, [1,2], 1)
        self.conv3x3_2 = ConvolutionBlock([32,64], [64,64], 3, [1,2], 1)
        self.conv3x3_3 = ConvolutionBlock([64,128], [128,128], 3, [1,2], 1)
        self.conv3x3_4 = ConvolutionBlock([128,256], [256,256], 3, [1,2], 1)  
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # global average pooling
        self.fc = nn.Linear(256, self.output_size)

        nn.init.xavier_uniform_(self.conv3x3_1.conv.weight)
        nn.init.xavier_uniform_(self.conv3x3_2.conv.weight)
        nn.init.xavier_uniform_(self.conv3x3_3.conv.weight)
        nn.init.xavier_uniform_(self.conv3x3_4.conv.weight)
        nn.init.zeros_(self.conv3x3_1.conv.bias)
        nn.init.zeros_(self.conv3x3_2.conv.bias)
        nn.init.zeros_(self.conv3x3_3.conv.bias)
        nn.init.zeros_(self.conv3x3_4.conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        

    def forward(self, x, action):
        x = x.unsqueeze(1)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv3x3_3(x)
        x = self.conv3x3_4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = gumbel_sigmoid(x, hard=True)
        if not self.interacting:
            x = torch.cat((x, action), 1)
        return x    

class MultiObjectEncoder(nn.Module):
    def __init__(self, soe1, soe2):
        super(MultiObjectEncoder).__init__()
        self.input_size = 2
        self.output_size = 1
        self.single_object_encoders = [soe1, soe2]
        for soe in self.single_object_encoders:
            soe.interacting = True
            for param in soe.parameters():  
                param.requires_grad = False
        self.conv3x3_1 = ConvolutionBlock([self.input_size,32], [32,32], 3, [1,2], 1)
        self.conv3x3_2 = ConvolutionBlock([32,64], [64,64], 3, [1,2], 1)
        self.conv3x3_3 = ConvolutionBlock([64,128], [128,128], 3, [1,2], 1)
        self.conv3x3_4 = ConvolutionBlock([128,256], [256,256], 3, [1,2], 1)  
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # global average pooling
        self.fc = nn.Linear(256, self.output_size)

        nn.init.xavier_uniform_(self.conv3x3_1.conv.weight)
        nn.init.xavier_uniform_(self.conv3x3_2.conv.weight)
        nn.init.xavier_uniform_(self.conv3x3_3.conv.weight)
        nn.init.xavier_uniform_(self.conv3x3_4.conv.weight)
        nn.init.zeros_(self.conv3x3_1.conv.bias)
        nn.init.zeros_(self.conv3x3_2.conv.bias)
        nn.init.zeros_(self.conv3x3_3.conv.bias)
        nn.init.zeros_(self.conv3x3_4.conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        

    def forward(self, x1,x2, action):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        y1 = self.single_object_encoders[0](x1, action)
        y2 = self.single_object_encoders[1](x2, action)
        x = torch.cat((x1, x2), 1)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv3x3_3(x)
        x = self.conv3x3_4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = gumbel_sigmoid(x, hard=True)
        x = torch.cat((y1, y2, x, action), 1)
        return x
        

class Encoder(nn.Module):
    def __init__(self, encoder_type, soe1=None, soe2=None):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        if encoder_type == "f1":
            self.encoder = SingleObjectEncoder(interacting=False)
        elif encoder_type == "f2":
            if soe1 is None or soe2 is None:
                raise ValueError("The encoder type f2 requires two single object encoders.")
            self.encoder = MultiObjectEncoder(soe1, soe2)
        else:
            raise ValueError(f"The encoder type {encoder_type} is not supported.")

    def forward(self, x, action):
        return self.encoder(x, action)


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels[0], out_channels[0], kernel_size, stride[0], padding)
        self.conv_2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size, stride[1], padding)
        self.bn = nn.BatchNorm2d(out_channels[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x