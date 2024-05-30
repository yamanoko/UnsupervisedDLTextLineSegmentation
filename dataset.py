import torch
from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
from torchvision.transforms import ToTensor
from PIL import Image

class CustomDataset(Dataset):
	def __init__(self, first_dir, second_dir, label):
		self.first_dir = first_dir
		self.second_dir = second_dir
		self.label = pd.read_csv(label, sep=' ')
		self.to_tensor = ToTensor()

	def __len__(self):
		return self.label.shape[0]

	def __getitem__(self, idx):
		first_img = Image.open(os.path.join(self.first_dir, self.label.iloc[idx, 0]))
		first_img = self.to_tensor(first_img)
		second_img = Image.open(os.path.join(self.second_dir, self.label.iloc[idx, 0]))
		second_img = self.to_tensor(second_img)
		binary_label = float(self.label.iloc[idx, 1] == self.label.iloc[idx, 2])
		return (first_img, second_img), torch.tensor([binary_label], dtype=torch.float32)