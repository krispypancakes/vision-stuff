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
  print('model size: {:.3f}MB'.format(size_all_mb))
  return size_all_mb

@torch.no_grad()
def estimate_loss(model, loader, crit):
  test_loss = []
  for x, y in loader:
    predictions = model(x)
    loss = crit(predictions, y)
    test_loss.append(loss.item())
  return np.mean(test_loss)
