import torch
from PIL import Image
import torch.nn as nn
from architecture import SegNet
import os
import argparse
from torch.utils.data import DataLoader
from dataset import BioImageDataset
from evaluation import *
import tqdm
from torchvision import transforms
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--weights_decay', type=float, default=3e-4)

args = parser.parse_args()


def train(num_epochs, epoch_old):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = 0.0
    for epoch in range(epoch_old, num_epochs):
        training_loss = 0.0
        acc = 0.0
        loop = tqdm.tqdm(train_loader)
        for batch_index, batch_samples in enumerate(loop):
            images = batch_samples['image'].to(device)
            labels = batch_samples['label'].to(device)
            optimizer.zero_grad()
            preds = net(images)
            preds = torch.sigmoid(preds)
            # print(preds[0])
            # exit()
            # preds_flat = outputs.view(outputs.size(0), -1)
            # labels_flat = labels.view(labels.size(0), -1)
            loss = criterion(preds, labels)
            acc += get_accuracy(preds, labels)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            loop.set_postfix(loss=loss.item())
        length = len(train_loader.dataset)
        # if best_acc < acc:
        #     best_acc = acc
        #     torch.save({
        #         'model_state_dict': net.state_dict(),
        #         'epoch': epoch,
        #         # 'scheduler_state_dict': scheduler.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, 'weights/best_ckp.pth')
        print(f'Done epoch {epoch + 1}/{num_epochs}: loss = {training_loss / length}, acc = {acc / len(train_loader)}')


if __name__ == '__main__':
    net = SegNet(1, 1).cuda()
    optimizer = torch.optim.Adam(
        params=net.parameters(),
        lr=args.learning_rate,
        betas=[args.beta1, args.beta2],
    )
    criterion = nn.CrossEntropyLoss()
    train_image = BioImageDataset(r'D:\research about semantic segmentation\u-net\data\membrane\train\aug')
    train_loader = DataLoader(train_image, batch_size=args.batch_size, shuffle=False, drop_last=False)

    train(5, 0)
