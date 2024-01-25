from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import numpy as np
from model.resnet import ResNet, BasicBlock, resnet34
from train import train, test
import os
from data import get_dataloader
import yaml


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# loading data
ap = argparse.ArgumentParser()
ap.add_argument("--protocol_file", type=str, default="/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/protocol.txt")
ap.add_argument("--dataset_dir", type=str, default="/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/")
ap.add_argument("--config_path", type=str, default="resnet_config.yaml")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--dev", action="store_true")
ap.add_argument("--eval", action="store_true")
args = ap.parse_args()

train_dataloader = get_dataloader(args.protocol_file, args.dataset_dir, args.batch_size, dev=args.dev, eval=args.eval)
dev_dataloader = get_dataloader(args.protocol_file, args.dataset_dir, args.batch_size, dev=True, eval=args.eval)
eval_dataloader = get_dataloader(args.protocol_file, args.dataset_dir, args.batch_size, dev=False, eval=True)


#load config
with open(args.config_path, 'r') as f_yaml:
    config = yaml.safe_load(f_yaml)


# build model
model = resnet34()
print("ResNet34")

model = torch.nn.DataParallel(model).cuda()

# define optimizer
optimizer_name = config['model']['optimizer']
lr = config['model']['lr']
momentum = config['model'].get('momentum', 0.9)  # Default value is 0.9

if optimizer_name.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_name.lower() == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
elif optimizer_name.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

best_dev_loss = np.inf

# load pre-trained model
checkpoint_path = os.path.join('./checkpoint', f"{config['model']['arc']}_lfcc.pth")
if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    print(f'Load pre-trained model of {config["model"]["arc"]}\n')
    print(state)
    best_dev_loss = state['acc']

# training with early stopping
epoch = config['model']['epoch']
epochs = config['model']['epochs']
iteration = config['model']['iteration']
patience = config['model']['patience']
log_interval = config['model']['log_interval']

print('\nStart training...')
while (epoch < epochs + 1) and (iteration < patience):
    train(train_dataloader, model, optimizer, epoch, True, log_interval)

    train_loss = test(train_dataloader, model, True, mode='Train loss')
    dev_loss = test(dev_dataloader, model, True, mode='dev loss')

    if dev_loss > best_dev_loss:
        iteration += 1
        print('\nLoss was not improved, iteration {0}\n'.format(str(iteration)))
    else:
        print(f'\nSaving model of {config["model"]["arc"]}\n')
        iteration = 0
        best_dev_loss = dev_loss
        state = {
            'net': model.module if hasattr(model, 'module') else model,
            'acc': dev_loss,
            'epoch': epoch,
        }
        checkpoint_path = os.path.join('./checkpoint', f"{config['model']['arc']}_lfcc_epoch{epoch}.pth")
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, checkpoint_path)

    epoch += 1

print('Finish Training!!')

#evaluate
eval_loss  = test(eval_dataloader, model, True, mode='Test loss')

print('Finished!!')