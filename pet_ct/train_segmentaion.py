import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from .dataload.seg_dataset import MyDataSet
from .seg_models.combine_model import Comebined
from .utils.segmentation_util import train_one_epoch, evaluate

def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    nw=4
    train_path = 'G:/pet-ct/train'
    test_path = 'G:/pet-ct/test'
    train_list = os.listdir(train_path)
    test_list = os.listdir(test_path)

    test_dataset = MyDataSet(test_path, test_list, transform=None)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4, )

    train_dataset = MyDataSet(train_path, train_list, transform=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw, )


    if os.path.exists("./weights") is False:
        os.makedirs("./weights")


    tb_writer = SummaryWriter()

    model = Comebined()
    model.to(device)

    param_dicts = [p for p in model.named_parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0.0000001, last_epoch=-1)

    best_loss = 100
    for epoch in range(args.epochs):

        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)

        test_loss = evaluate(model=model,
                             data_loader=test_loader,
                             device=device,
                             epoch=epoch)

        tags = ["train_loss", "test_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], test_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        print('训练集loss: ', train_loss)
        print('测试集loss: ', test_loss, '学习率: ', optimizer.param_groups[0]["lr"])
        scheduler.step()

        if test_loss < best_loss:
            torch.save(model, "./weights/best_loss.pth")
            best_loss = test_loss

        torch.save(model, "./weights/last.pth")


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # batch_size 改成了3
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=201, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    opt = parser.parse_args()

    main(opt)
