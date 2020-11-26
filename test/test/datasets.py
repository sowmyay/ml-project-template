import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):
	def __init__(self, paths, transform=None):
		super().__init__()
		self.paths = paths
		self.transform = transform
	
	def __len__(self):
		return len(self.paths)

	def __getitem__(self, i):
		path = self.paths[i]
		item = np.load(path, allow_pickle=False)
		target = True
		if self.transform:
			item = self.transform(item)
		return item, target
