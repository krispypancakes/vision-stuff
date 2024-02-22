# for tuning:
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pickle
from tqdm import trange
from helpers import estimate_loss, normalize_tensor


class CiFaData(Dataset):
  def __init__(self, stage="train", transform=None, device="cpu"):
    # self.device = device
    self.base_folder = "cifar-10-batches-py"
    self.transform = transform
    if stage == "train":
      batch_collection = [f"data_batch_{i}" for i in range(1, 5)]
    elif stage == "val":
      batch_collection = ["data_batch_5"]
    elif stage == "test":
      batch_collection = ["test_batch"]
    else:
      raise ValueError("Invalid stage, choose from train, val, test.")
    self.x_data, self.y_data = [], []
    for batch in batch_collection:
      with open(f"../data/cifar-10-batches-py/{batch}", "rb") as f:
        data = pickle.load(f, encoding="latin1") 
        self.x_data.extend(data["data"])
        self.y_data.extend(data["labels"])
    self.y_data = torch.tensor(self.y_data)
    self.x_data = normalize_tensor(torch.tensor(np.vstack(self.x_data).reshape(-1, 3, 32, 32))) # from list to vstack; results in (N, 3, 32, 32)
  def __len__(self):
    return self.y_data.shape[0]
  def __getitem__(self, idx):
    if self.transform:
      return self.transform(self.x_data[idx]), self.y_data[idx]
    return self.x_data[idx], self.y_data[idx]

class PrepBlock(nn.Module):
  # fixed channels for cifar
  def __init__(self):
    super().__init__()
    self.prep_block = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
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
      nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=stride),
      nn.BatchNorm2d(outchannels),
      nn.ReLU(),
      nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=1),
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
  
class ResNet18New(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = nn.Sequential(
      PrepBlock(),
      ComputeBlock(64, 64, stride=1),
      ComputeBlock(64,128, stride=2),
      ComputeBlock(128,256, stride=2),
      ComputeBlock(256, 512, stride=2),
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Flatten(start_dim=1),
      nn.Linear(512, 10)
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
        module.bias.data.zero_() 

def train(config, device):
  model = ResNet18New()
  model.to(device)

  tf = transforms.Compose([transforms.RandomResizedCrop((32,32)), 
                         transforms.RandomHorizontalFlip(p=0.58), 
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_ds = CiFaData(stage="train", device=device, transform=tf)
  val_ds = CiFaData(stage="val", device=device)
  train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True) 
  val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=['mom'], weight_decay=config['wd'])

  checkpoint = session.get_checkpoint() 

  if checkpoint:
    checkpoint_state = checkpoint.to_dict()
    start_epoch = checkpoint_state['epoch']
    model.load_state_dict(checkpoint_state['model_state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
  else:
    start_epoch = 0
  
  for epoch in (t:=trange(start_epoch, 20)):
    running_loss = 0.0
    epoch_steps = 0
    for i, data in enumerate(train_loader):
      x, y = data
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad(set_to_none=True)
      out = model(x)
      loss = criterion(out, y)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      epoch_steps +=1
      if i % 2000 == 1999:
        t.set_description(f'epoch: {epoch+1} | step: {i+1} | training_loss: {running_loss/epoch_steps}')
        # why set it to zero again ?
        running_loss = 0.0

    val_loss, val_acc = estimate_loss(model, val_loader, criterion, device)
    checkpoint_data = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint = Checkpoint.from_dict(checkpoint_data)
    session.report(
      {'loss': val_loss, 'accuracy': val_acc},
      checkpoint=checkpoint
    )
  print('training done')


def main(max_num_epochs, num_samples=5):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  config = {
  "wd": tune.loguniform(1e-3, 1e-2),
  "mom": tune.loguniform(0.5, 0.9),
  "lr": tune.loguniform(1e-4, 1e-1),
  "batch_size": tune.choice([64, 128, 256, 512, 1024])
    }
  
  scheduler = ASHAScheduler(
    metric='loss',
    mode='min',
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2
  )
  tuner = tune.Tuner(
    tune.with_resources(
      tune.with_parameters(train),
      resources={'cpu':4, 'gpu':0.25}
    ),
    tune_config=tune.TuneConfig(
      metric='loss',
      mode='min',
      scheduler=scheduler,
      num_samples=num_samples
    ), 
    param_space={'config':config, 'device':device}
  )
  
  # tuner = tune.Tuner(
  #   partial(train),
  #   resources_per_trial={'cpu': 4, 'gpu':0.25},
  #   config=config,
  #   device=device,
  #   num_samples=10,
  #   scheduler=scheduler
  # )
  results = tuner.fit()
  best_trial = results.get_best_result('loss', 'min')
  print(f"Best trial config: {best_trial.config}")
  print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
  print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

  best_trained_model = ResNet18New()
  best_trained_model.to(device)
  best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
  best_checkpoint_data = best_checkpoint.to_dict()
  best_trained_model.load_state_dict(best_checkpoint_data['model_state_dict'])
  
  test_ds = CiFaData(stage="test", device=device) 
  test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=12, pin_memory=True)
   
  test_loss, test_acc = estimate_loss(best_trained_model, test_loader, nn.CrossEntropyLoss(), device)
  print(f"Best trial test set loss: {test_loss} | test acc: {test_acc}")
  
  
if __name__ == "__main__":
  main(20)
  