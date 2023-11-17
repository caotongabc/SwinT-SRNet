import os
import argparse

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate
from pytorchtools import EarlyStopping


def collate_fn(batch):
    images, labels = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return images, labels


def main(args, indexForIter):
    # seed
    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tb_writer = SummaryWriter('./logs/Tiny{}/'.format(indexForIter + 1))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    Mean = [0.78772519, 0.69126706, 0.73102253]
    Std = [0.11292258, 0.16699339, 0.12156582]
    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(int(img_size * 1.243)),
            # transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
            # transforms.ColorJitter(contrast=0.3),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAdjustSharpness(1),
            transforms.ToTensor(),
            transforms.Normalize(Mean, Std)]),

        "val": transforms.Compose([
            # transforms.Resize(img_size),
            transforms.Resize(int(img_size * 1.243)),
            transforms.CenterCrop(img_size),
            # transforms.ColorJitter(contrast=0.3),
            transforms.ToTensor(),
            # transforms.RandomAdjustSharpness(1),
            transforms.Normalize(Mean, Std)])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    data_train = datasets.ImageFolder(
        '../../data_set/flower_data/x2/train{}zhe'.format(indexForIter + 1),
        transform=data_transform["train"])
    print("data_train" + data_train.root)
    val_dataset = datasets.ImageFolder(
        '../../data_set/flower_data/x2/val{}zhe'.format(indexForIter + 1),
        transform=data_transform["val"])
    print("val_dataset" + val_dataset.root)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw
                                             )

    model = create_model(num_classes=args.num_classes, drop_rate=0.15).to(device)
    # print(model)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()

            self.avg_pool = nn.AvgPool2d(7)

            self.upsample = nn.Upsample(scale_factor=7, mode='nearest')

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)

            self.conv3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

        def forward(self, x):
            x_avg = self.avg_pool(x)

            x_up = self.upsample(x_avg)

            x_high_freq = x - x_up

            x1 = self.conv1(x_high_freq)

            x3 = self.conv3(x1)

            x4 = x3 + x

            return x4

    cusmodel = MyModel()
    # print(cusmodel)
    combined_model = nn.Sequential(cusmodel, model).to(device)
    pg = [p for p in combined_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    scheduler = CosineAnnealingLR(optimizer, 160)
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=combined_model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=combined_model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        # change model and set various lr
        desc = "bestmodel"
        early_stopping(val_acc, combined_model, epoch, indexForIter + 1, desc)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
    tb_writer.close()


if __name__ == '__main__':
    for indexForIter in range(5):
        indexForIter = indexForIter + 0
        print("{}fold train".format(indexForIter + 1))
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_classes', type=int, default=8)
        parser.add_argument('--epochs', type=int, default=160)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--data-path', type=str,
                            default=" ".format(indexForIter + 1))
        parser.add_argument('--weights', type=str, default='',
                            help='initial weights path')
        parser.add_argument('--freeze-layers', type=bool, default=False)
        parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
        early_stopping = EarlyStopping(20, verbose=True)
        opt = parser.parse_args()
        main(opt, indexForIter)
