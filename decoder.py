import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, decoder_type):
        super(Decoder, self).__init__()
        self.output_size = 3 if decoder_type == "f1" else 6
        self.fc_1 = nn.Linear(5, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, self.output_size)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x
