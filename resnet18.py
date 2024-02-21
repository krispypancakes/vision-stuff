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
from helpers import get_model_size, estimate_loss, normalize_tensor


class CiFaData(Dataset):
  def __init__(self, stage="train", transform=None, device="cpu"):
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
    self.x_data = []
    self.y_data = []
    for batch in batch_collection:
      with open(f"data/cifar-10-batches-py/{batch}", "rb") as f:
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

class ComputeBlock(nn.Module):
  def __init__(self, inchannels, outchannels, stride):
    super().__init__()
    self.compute_block = nn.Sequential(
      nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=stride),
      nn.BatchNorm2d(outchannels),
      nn.ReLU(),
      nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, padding=1, stride=1),
      nn.BatchNorm2d(outchannels)     
    )
  def forward(self, x):
    return self.compute_block(x)

class ResidualBlock(nn.Module):
  # this is where the dimension matching happens if necessary 
  def __init__(self, inchannels, outchannels):
    super().__init__()
    self.res_block = nn.Sequential(
      nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=2),
      nn.BatchNorm2d(outchannels)
    )
  def forward(self, x):
    return self.res_block(x)
  
class ResNet18(nn.Module):
  def __init__(self):
    super().__init__()
    self.prep_block = PrepBlock()
    self.compute_block1 = ComputeBlock(64, 64, stride=1)
    self.relu1 = nn.ReLU()
    self.res_block2 = ResidualBlock(64,128)
    self.compute_block2 = ComputeBlock(64,128, stride=2)
    self.relu2 = nn.ReLU()
    self.res_block3 = ResidualBlock(128,256)
    self.compute_block3 = ComputeBlock(128, 256, stride=2)
    self.relu3 = nn.ReLU()
    self.res_block4 = ResidualBlock(256, 512)
    self.compute_block4 = ComputeBlock(256, 512, stride=2)
    self.relu4 = nn.ReLU()
    self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512, 10)
  def forward(self, x):
    # -------- prep block --------
    x_skip = self.prep_block(x) # (B,64,8,8)
    # ---------- Block1 ---------- 
    x = self.compute_block1(x_skip) # (B,64,8,8)
    x_skip = self.relu1(x+x_skip)
    # ---------- Block2 ---------- 
    x_skip = self.res_block2(x_skip) # (B,128,4,4)
    x = self.compute_block2(x)
    x = self.relu2(x+x_skip)
    # ---------- Block3 ---------- 
    x_skip = self.res_block3(x) # (B,256,2,2)
    x = self.compute_block3(x)
    x = self.relu3(x+x_skip)
    # ---------- Block4 ---------- 
    x_skip = self.res_block4(x) # (B,512,1,1)
    x = self.compute_block4(x)
    x = self.relu4(x+x_skip)
    # ---------- Linear ----------
    x = self.avg_pool(x)
    x = torch.flatten(x, start_dim=1) # (B,512)
    return self.fc(x) # (B,10)


def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  epochs = 300
  lr = 0.001

  train_ds = CiFaData(stage="train", device=device, transform=transforms.RandomRotation(degrees=(0, 180)))
  val_ds = CiFaData(stage="val", device=device)
  test_ds = CiFaData(stage="test", device=device)

  train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True, num_workers=14) 
  val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=True, num_workers=14)
  test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True, num_workers=14)

  model = ResNet18()
  model.to(device)

  optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad==True], lr=lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=0)
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
    val_losses.append(estimate_loss(model, val_loader, criterion, device))
    scheduler.step(metrics=val_losses[-1])
    t.set_description(f"epoch {i+1} | training loss: {losses[-1]:.4f} | validation loss: {val_losses[-1]:.4f}")
  duration = time.time() - time1
  test_loss = estimate_loss(model, test_loader, criterion, device) 
  print(f'final test loss is : {test_loss}')
  print(f'this took {duration / 60:.4f} minutes for training')

  # TODO: print results of training run or add tensorboard / mlflow tracking
if __name__ == "__main__":
  main()
