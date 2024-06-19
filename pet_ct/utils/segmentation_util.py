import math
import torch
from torch.nn import functional as F
import tqdm
import sys
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def criterion(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    data_loader = tqdm(data_loader, file=sys.stdout)
    optimizer.zero_grad()
    loss_value = 0
    num_samples = 0
    for step, data in enumerate(data_loader):
        image1, image2, target = data
        num_samples += image1.shape[0]
        image1, image2, target = image1.to(device), image2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image1, image2)
        output = F.softmax(output, dim=1).float()
        target = F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float()
        loss = criterion(output, target, multiclass=True)
        loss.backward()
        loss_value += loss.item()*image1.shape[0]

        data_loader.desc = "[Train epoch {}] loss: {:.3f}".format(
            epoch, loss.item())

    return loss_value/num_samples

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.train()
    data_loader = tqdm(data_loader, file=sys.stdout)
    loss_value = 0
    num_samples = 0
    for step, data in enumerate(data_loader):
        image1, image2, target = data
        num_samples += image1.shape[0]
        image1, image2, target = image1.to(device), image2.to(device), target.to(device)
        output = model(image1, image2)
        output = F.softmax(output, dim=1).float()
        target = F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float()
        loss = criterion(output, target, multiclass=True)
        loss_value += loss.item()*image1.shape[0]

        data_loader.desc = "[Test epoch {}] loss: {:.3f}".format(
            epoch, loss.item())

    return loss_value/num_samples