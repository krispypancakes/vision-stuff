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
def estimate_loss(model, loader, crit):
  # torch.no_grad tells torch that there there's no grad -> tensors
  # model.eval() takes care of batchnorm or dropout
  model.eval()
  test_loss = []
  for x, y in loader:
    predictions = model(x)
    loss = crit(predictions, y)
    test_loss.append(loss.item())
  # put it back in training mode
  model.train()
  return np.mean(test_loss)

def normalize_tensor(tensor, _torch=True):
  if type(tensor) == torch.tensor:
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
  else:
    # what we do with tiny tensors
    min_val = tensor.min()
    max_val = tensor.max()
  return (tensor - min_val) / (max_val - min_val)
