import os
import json
import sys

import torch
from PIL import Image
from prettytable import PrettyTable
from torch import nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from model import swin_tiny_patch4_window7_224 as create_model

from utils import testmodel
from torch.utils.tensorboard import SummaryWriter


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Mean = [0.78772519, 0.69126706, 0.73102253]
    Std = [0.11292258, 0.16699339, 0.12156582]
    img_size = 224
    averageACC = 0.0
    allMatrix = np.zeros((8, 8))
    for index in range(5):
        index = index + 0
        print("For {} fold Test".format(index + 1))
        data_transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.243)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(Mean, Std)])

        # 实例化验证数据集
        path = '../../data_set/testData/x2/pollen{}'.format(index + 1)
        print("testData：" + path)
        testset = datasets.ImageFolder(path, transform=data_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, drop_last=False)

        # read class_indict
        json_path = 'class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = create_model(num_classes=8, drop_rate=0.2).to(device)
        # load model weights
        writer = SummaryWriter('../Testlog')
        bestAcc = 0
        for epoch in range(1):
            model_weight_path = ''.format(index + 1)

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

            combined_model = nn.Sequential(cusmodel, model).to(device)
            combined_model.load_state_dict(torch.load(model_weight_path, map_location=device))
            combined_model.eval()
            val_loss, val_acc, matrixTem = testmodel(model=combined_model,
                                                     data_loader=testloader,
                                                     device=device,
                                                     epoch=epoch)
            allMatrix = matrixTem + allMatrix
            writer.add_scalar("testAcc", val_acc, epoch)
            averageACC = averageACC + val_acc
        writer.close()
    print("Five fold Accuracy：{}".format(averageACC / 5))

    def summary():
        # calculate accuracy
        sum_TP = 0
        matrix = allMatrix
        labels = pollenlist = [
            "artemisia",
            "chenopodiaceae",
            "cupressaceae",
            "gramineae",
            "moraceae",
            "pinaceae",
            "salicaceae_populus",
            "salicaceae_salix"
        ]
        for i in range(8):
            sum_TP += matrix[i, i]
        acc = sum_TP / np.sum(matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1_score"]
        for i in range(8):
            TP = matrix[i, i]
            FP = np.sum(matrix[i, :]) - TP
            FN = np.sum(matrix[:, i]) - TP
            TN = np.sum(matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1_score = round((2 * Precision * Recall) / (Precision + Recall), 3)
            table.add_row([labels[i], Precision, Recall, Specificity, F1_score])
        print("fivefold data")
        print(matrix)
        print("fivefold predict data")
        print(table)

        return matrix

    # self = object
    summary()

    allMatrix = allMatrix.astype(int)
    print(allMatrix)


if __name__ == '__main__':
    main()
