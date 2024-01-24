from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable

def train(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
    """
    This function is used to train the model.
    """
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.item()
        if verbose:
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                      f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return global_epoch_loss / len(loader.dataset)

def test(loader, model, cuda, mode, verbose=True):
    """
    This function is used to test the model.
    """
    model.eval()
    xx_loss = 0
    correct = 0
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        xx_loss += F.nll_loss(output, target, reduction='sum').item()   # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cuda().sum()

    xx_loss /= len(loader.dataset)
    if verbose:
        print(f'{mode} set: Average loss: {xx_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} '
              f'({100. * correct / len(loader.dataset):.0f}%)')

    return xx_loss