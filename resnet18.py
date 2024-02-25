import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pickle
from tqdm import trange
import matplotlib.pyplot as plt
import time
from helpers import get_model_size, estimate_loss, CiFaData


class PrepBlock(nn.Module):
  # fixed channels for cifar
  def __init__(self):
    super().__init__()
    self.prep_block = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
  def forward(self, x):
    return self.prep_block(x)
  def init_weights(self):
    for layer in self.prep_block:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
          layer.bias.data.zero_()

class ComputeBlock(nn.Module):
  def __init__(self, inchannels, outchannels, stride, downsample=None):
    super().__init__()
    self.convblock = nn.Sequential(
      nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=stride, bias=False),
      nn.BatchNorm2d(outchannels),
      nn.ReLU(),
      nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=1, bias=False),
      nn.BatchNorm2d(outchannels)
    )
    self.downsample = downsample
    if not inchannels == outchannels:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(outchannels)
      )
    self.relu = nn.ReLU()
  def forward(self, x):
    x_skip = x 
    x = self.convblock(x)
    if self.downsample:
      x_skip = self.downsample(x_skip)
    out = self.relu(x+x_skip)
    return out
  def init_weights(self):
    for layer in self.convblock:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
          layer.bias.data.zero_()
    if self.downsample:
      for layer in self.downsample:
        if isinstance(layer, nn.Conv2d):
          nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
          if layer.bias is not None:
            layer.bias.data.zero_()
  
  
class ResNet18(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = nn.Sequential(
      PrepBlock(), # (B,64,8,8)
      ComputeBlock(64, 64, stride=1), # (B,64,8,8)
      ComputeBlock(64,128, stride=2), # (B,128,4,4)
      ComputeBlock(128,256, stride=2), # (B,256,2,2)
      ComputeBlock(256, 512, stride=2), # (B,512,1,1)
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Flatten(start_dim=1), # (B,512)
      nn.Linear(512, 10) # (B, 10)
    )
  def forward(self, x):
    return self.resnet(x)
  def init_weights(self):
    for module in self.modules():
      if isinstance(module, ComputeBlock):
        module.init_weights()
      elif isinstance(module, PrepBlock):
        module.init_weights()
      elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  epochs = 900
  batch_size = 256
  momentum = 0.875
  w_decay = 0.00125
  n_worksers = 14
  lr = 0.001
  tf = transforms.Compose([transforms.RandomResizedCrop((32,32)), 
                         transforms.RandomHorizontalFlip(p=0.5)])
  # mean and std of channels of cifar
  params = torch.tensor([0.4919, 0.4827, 0.4472]), torch.tensor([0.2470, 0.2434, 0.2616])
  dataset_path = "data/"
  train_ds = CiFaData(stage="train", path=dataset_path, transform=tf, dataset_params=params)
  val_ds = CiFaData(stage="val", path=dataset_path, dataset_params=params)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_worksers, pin_memory=True) 
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_worksers, pin_memory=True)

  model = ResNet18()
  model.to(device)

  optimizer = optim.SGD(params=[p for p in model.parameters() if p.requires_grad==True], lr=lr, momentum=momentum, weight_decay=w_decay)
  # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=0)
  criterion = nn.CrossEntropyLoss()

  get_model_size(model)

  # training loop
  losses = []
  raw_losses = []
  val_losses = []
  time1 = time.time()
  for i in (t:=trange(epochs)):
    epoch_loss = []
    for x, y in train_loader:
      predictions = model(x.to(device))
      loss = criterion(predictions, y.to(device))
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      raw_losses.append(loss.item())
      epoch_loss.append(loss.item())
    losses.append(np.mean(epoch_loss))
    val_loss, val_acc = estimate_loss(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    # scheduler.step(metrics=val_losses[-1])
    t.set_description(f"epoch {i+1} | training loss: {losses[-1]:.4f} | validation loss: {val_losses[-1]:.4f}")
  duration = time.time() - time1
  # test_loss = estimate_loss(model, test_loader, criterion, device) 
  print(f'final validation loss is : {val_loss}')
  print(f'this took {duration / 60:.4f} minutes for training')
  # store results; just pickle it for now.
  with open('training_run.pickle', 'wb') as f:
    pickle.dump((losses, val_losses), f)

if __name__ == "__main__":
  main()
