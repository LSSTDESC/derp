import argparse
import numpy as np
import os
import torch
from data.dataloader import DerpData
from torch.utils.data import DataLoader
from net.regressor import Regressor
from optim_utils.loss_functions import reg_bce_loss

##########
# Config #
##########

# Read in command-line args
parser = argparse.ArgumentParser(description='DerpData Training')
parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 30)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='N',
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# Set device
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device=='cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

############
# Data I/O #
############

# Data paths
X_path = os.path.join('data', 'X.csv')
y_path = os.path.join('data', 'y.csv')

# Define dataset
train_data = DerpData(X_filepath=X_path, y_filepath=y_path, is_train=True)
val_data = DerpData(X_filepath=X_path, y_filepath=y_path, is_train=True)

# Define dataloader
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, **kwargs)

#########
# Model #
#########

model = Regressor(input_dim=train_data.X_dim, output_dim=train_data.y_dim)
model = model.to(device)

################
# Optimization #
################

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True, weight_decay=0.1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
loss_function = reg_bce_loss

#################
# Train and val #
#################

def train(epoch):
    """ Runs an epoch of training
    Parameters
    ----------
    epoch: int
        Index of current epoch
    """
    model.train()
    lr_scheduler.step()
    train_loss = 0
    for batch_idx, (X_, y_, bin_y_, mask_X_, mask_y_, mask_bin_y_) in enumerate(train_loader):
        X_ = X_.float().to(device)
        y_ = y_.float().to(device)
        bin_y_ = bin_y_.float().to(device)
        optimizer.zero_grad()
        output = model(X_)
        loss = loss_function(output, y_, bin_y_, mask_y_, mask_bin_y_, scale_bce=0.1)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                epoch + 1, 
                batch_idx * len(X_), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(X_)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch + 1, 
        train_loss/len(train_loader.dataset)))
    return train_loss/len(train_loader.dataset)

def val(epoch):
    """ Runs an epoch of validation
    Parameters
    ----------
    epoch: int
        Index of current epoch
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (X_, y_, bin_y_, mask_X_, mask_y_, mask_bin_y_)  in enumerate(val_loader):
            X_ = X_.float().to(device)
            y_ = y_.float().to(device)
            bin_y_ = bin_y_.float().to(device)
            output = model(X_)
            val_loss += loss_function(output, y_, bin_y_, mask_y_, mask_bin_y_, scale_bce=0.1).item()
    print('====> Validation set loss: {:.4f}'.format(val_loss/len(val_loader.dataset)))
    return val_loss/len(val_loader.dataset)

def main():
    """ Runs the experiment combining train and validation
    
    """
    losses = {}
    losses['train'] = np.zeros(args.epochs)
    losses['val'] = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        losses['train'][epoch] = train(epoch)
        losses['val'][epoch] = val(epoch)
    torch.save(model.state_dict(), 'checkpoint/regressor_%d_%f.6.pth' %(args.epochs, args.learning_rate)) 
    np.save('train_losses', losses['train'])
    np.save('val_losses', losses['val'])

if __name__ == "__main__":
    main()
