# for tuning:
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import tempfile
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import ResNet18
from tqdm import trange
from helpers import estimate_loss, CiFaData


def training(config, params, dataset_path, device='cuda'):
  model = ResNet18()
  model.to(device)

  tf = transforms.Compose([transforms.RandomResizedCrop((32,32)), 
                         transforms.RandomHorizontalFlip(p=0.5)])

  train_ds = CiFaData(stage="train", transform=tf, dataset_params=params, path=dataset_path)
  val_ds = CiFaData(stage="val", dataset_params=params, path=dataset_path)
  train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True) 
  val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['mom'], weight_decay=config['wd'])

  if train.get_checkpoint():
    loaded_checkpoint = train.get_checkpoint()
    with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
      checkpoint = torch.load(os.path.join(loaded_checkpoint_dir, 'checkpoint.pt'))
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  else:
    start_epoch = 0 
  
  for epoch in (t:=trange(start_epoch, 50)):
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
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
      path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')
      torch.save({'model_state_dict': model.state_dict(), 'opti_state_dict': optimizer.state_dict(), 'epoch': epoch}, path)
      checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
      train.report({'loss': val_loss, 'accuracy': val_acc}, checkpoint=checkpoint)
    
  print('training done')

def main(max_num_epochs, num_samples=5):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  params = torch.tensor([0.4919, 0.4827, 0.4472]), torch.tensor([0.2470, 0.2434, 0.2616])
  dataset_path = "/home/pt/hacking/vision-stuff/data/"
  config = {
  "wd": tune.loguniform(1e-3, 1e-2),
  "mom": tune.loguniform(0.5, 0.9),
  "lr": tune.loguniform(1e-4, 1e-1),
  "batch_size": tune.choice([128, 256, 512])
    }
  
  scheduler = ASHAScheduler(
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2
  )
  tuner = tune.Tuner(
    tune.with_resources(tune.with_parameters(training, params=params, dataset_path=dataset_path),
                        resources={'cpu':4, 'gpu':0.25}),
    tune_config=tune.TuneConfig(metric='loss', mode='min', scheduler=scheduler, num_samples=num_samples), 
    param_space=config
  )
  
  results = tuner.fit()
  best_trial = results.get_best_result('loss', 'min')
  print(f"Best trial config: {best_trial.config}")
  print(f"Best trial final validation loss: {best_trial.metrics['loss']}")
  print(f"Best trial final validation accuracy: {best_trial.metrics['accuracy']}")

  best_trained_model = ResNet18()
  best_trained_model.to(device)
  checkpoint_path = os.path.join(best_trial.checkpoint.to_directory(), 'checkpoint.pt')
  checkpoint = torch.load(checkpoint_path)
  best_trained_model.load_state_dict(checkpoint['model_state_dict'])
  

  
  val_ds = CiFaData(stage="val", dataset_params=params, path=dataset_path)
  val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
  val_loss, val_acc = estimate_loss(best_trained_model, val_loader, nn.CrossEntropyLoss(), device=device)
  print(f"Best trial test set loss: {val_loss} | test acc: {val_acc}")
  
  
if __name__ == "__main__":
  main(40)
  