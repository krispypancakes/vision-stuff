import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pickle
from tqdm import trange
import time
from helpers import get_model_size, estimate_loss, CiFaData
from models import ResNet18


def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  epochs = 900
  batch_size = 256
  momentum = 0.54
  w_decay = 0.00472
  n_worksers = 14
  lr = 0.0027
  tf = transforms.Compose([transforms.RandomResizedCrop((32,32)), 
                         transforms.RandomHorizontalFlip(p=0.5),
                         ])
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
  print(f'final validation loss is : {val_loss} | final validation accuracy is : {val_acc}')
  print(f'this took {duration / 60:.4f} minutes for training')
  # store results; just pickle it for now.
  with open('training_run.pickle', 'wb') as f:
    pickle.dump((losses, val_losses), f)

if __name__ == "__main__":
  main()
