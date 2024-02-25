""" useful helpers """
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


class CiFaData(Dataset):
  def __init__(self, path, dataset_params=None, stage="train", transform=None):
    self.base_folder = "cifar-10-batches-py"
    self.dataset_params = dataset_params
    self.transform = transform
    if stage == "all":
      batch_collection = [f"data_batch_{i}" for i in range(1, 6)]
      batch_collection.append("test_batch")
    elif stage == "train":
      batch_collection = [f"data_batch_{i}" for i in range(1, 5)]
    elif stage == "val":
      batch_collection = ["data_batch_5"]
    elif stage == "test":
      batch_collection = ["test_batch"]
    else:
      raise ValueError("Invalid stage, choose from all, train, val, test.")
    self.x_data = []
    self.y_data = []
    for batch in batch_collection:
      with open(path + self.base_folder + '/' + batch, 'rb') as f:
        data = pickle.load(f, encoding="latin1") 
        self.x_data.extend(data["data"])
        self.y_data.extend(data["labels"])
    self.y_data = torch.tensor(self.y_data)
    if self.dataset_params is not None:
      self.x_data = normalize_tensor(torch.tensor(np.vstack(self.x_data).reshape(-1, 3, 32, 32)), self.dataset_params) # from list to vstack; results in (N, 3, 32, 32)
    else:
      self.x_data = torch.tensor(np.vstack(self.x_data).reshape(-1, 3, 32, 32))
  def __len__(self):
    return self.y_data.shape[0]
  def __getitem__(self, idx):
    if self.transform:
      return self.transform(self.x_data[idx]), self.y_data[idx]
    return self.x_data[idx], self.y_data[idx]

def get_model_size(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()
  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f} MB'.format(size_all_mb))
  return size_all_mb

@torch.no_grad()
def estimate_loss(model, loader, crit, device):
  # torch.no_grad tells torch that there there's no grad -> tensors
  # model.eval() takes care of batchnorm or dropout
  model.eval()
  correct, total_loss, total_count = 0, 0, 0
  for x, y in loader:
    x, y = x.to(device), y.to(device)
    predictions = model(x)
    loss = crit(predictions, y)
    total_loss += loss.item() * x.shape[0]
    total_count += x.shape[0]
    correct += (predictions.argmax(dim=1) == y).sum().item()
  return total_loss/total_count, correct/total_count

def get_parameters(loader):
  data = None
  print('loading the dataset')
  for x, _ in loader:
    data = torch.cat([data, x.float()]) if data is not None else x.float()
  print('calculate the parameters')
  mean_r = data[:,0,:,:].mean()
  mean_g = data[:,1,:,:].mean()
  mean_b = data[:,2,:,:].mean()
  std_r = data[:,0,:,:].std()
  std_g = data[:,1,:,:].std()
  std_b = data[:,2,:,:].std()
  params = (torch.tensor([x/255.0 for x in [mean_r, mean_g, mean_b]]), torch.tensor([x/255.0 for x in [std_r, std_g, std_b]]))
  return params
    
def normalize_tensor(tensor, params):
  """ normalizes and standardizes tensor using (normalized) mean, std of specifig dataset """
  _mean, _std = params
  _mean_tensor = _mean.reshape(1, 3, 1, 1)
  _std_tensor = _std.reshape(1, 3, 1, 1)
  norm_tensor = (tensor - _mean_tensor) / _std_tensor
  return norm_tensor
