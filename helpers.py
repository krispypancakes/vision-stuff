""" useful helpers """
import torch
import numpy as np

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
