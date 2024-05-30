import torch
from torch import nn

from cnn import ResNet

class PretrainModel(nn.Module):
	def __init__(self, num_features=256):
		super(PretrainModel, self).__init__()
		self.cnn = ResNet(num_features=num_features)
		self.fc1 = nn.Linear(num_features*2, num_features*2)
		self.fc2 = nn.Linear(num_features*2, num_features)
		self.fc3 = nn.Linear(num_features, 1)
	
	def forward(self, x1, x2):
		out1 = self.cnn(x1)
		out2 = self.cnn(x2)
		out = torch.cat((out1, out2), dim=1)
		out = torch.relu(self.fc1(out))
		out = torch.relu(self.fc2(out))
		out = self.fc3(out)
		return out
