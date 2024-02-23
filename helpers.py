""" useful helpers """
import torch


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

def normalize_tensor(tensor, _torch=True):
  if type(tensor) == torch.tensor:
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
  else:
    # what we do with tiny tensors
    min_val = tensor.min()
    max_val = tensor.max()
  return (tensor - min_val) / (max_val - min_val)
