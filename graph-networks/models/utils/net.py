from utils.layer import GCNLayer
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, features_size, labels_num, neuro_num):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(features_size, neuro_num)
        self.layer2 = GCNLayer(neuro_num, labels_num)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x