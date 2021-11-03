from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm


class Trainer:
  def __init__(self, root='./'):
    self.reset_data()
    self.root = Path(root)

  def reset_data(self):
    self.loss_train = list()
    self.loss_test = list()
    self.lrs = list()
    self.best_loss_test = float('inf')
    
  def training_loop(self,
                    model,
                    trainloader,
                    testloader,
                    opt,
                    loss_fn, 
                    device,
                    epochs, 
                    writer=None, 
                    save_model=True, 
                    reset_data=True, 
                    scheduler=None):
    scheduler_state_dict = None
    if reset_data:
      self.reset_data()
    model.to(device)
    for epoch in tqdm(range(1, epochs + 1)):
      model, running_loss_train = self.train_nn(model,
                                                trainloader,
                                                opt,
                                                loss_fn,
                                                device)
      with torch.no_grad():
        running_loss_test = self.eval_nn(model, 
                                         testloader,
                                         loss_fn,
                                         device)    
      
      curr_loss_train = np.mean(running_loss_train)
      curr_loss_test = np.mean(running_loss_test)
      curr_lr = opt.param_groups[0]["lr"]

      self.loss_train.append(curr_loss_train)
      self.loss_test.append(curr_loss_test)
      self.lrs.append(curr_lr)

      if scheduler is not None:
        scheduler.step()
        scheduler_state_dict = scheduler.state_dict()
      
      if writer:
        writer.add_scalar('Loss/train', curr_loss_train, epoch)
        writer.add_scalar('Loss/test', curr_loss_test, epoch)
        writer.add_scalar('Learning rate', curr_lr, epoch)

      if (epoch) % 5 == 0:
        print(f"Epoch [{epoch}/{epochs}]\n\
\tloss TRAIN: {curr_loss_train:.4f},\tloss TEST: {curr_loss_test:.4f}") 

      if save_model:    
        if curr_loss_test < self.best_loss_test:
          torch.save({'epoch': epoch, 
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': opt.state_dict(),
                      'scheduler_state_dict': scheduler_state_dict,
                      'loss': curr_loss_test}, self.root / 'best_model.pth')
          self.best_loss_test = curr_loss_test
    
  def train_nn(self, model, trainloader, opt, loss_fn, device):
    running_loss = list()
    
    model.train()
    for inputs, _ in trainloader:
      inputs = inputs.to(device)
      
      opt.zero_grad()
      outputs, _ = model(inputs)
      loss = loss_fn(outputs, inputs)      
      loss.backward()
      opt.step()
      
      running_loss.append(loss.cpu().item())
    return model, running_loss
  
  def eval_nn(self, model, testloader, loss_fn, device):
    running_loss = list()
    
    model.eval()   
    for inputs, _ in testloader:
      inputs = inputs.to(device)
      outputs, _ = model(inputs)
      loss = loss_fn(outputs, inputs)
      running_loss.append(loss.cpu().item())
    return running_loss