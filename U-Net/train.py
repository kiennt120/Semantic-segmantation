import torch
from PIL import Image
from u_net import UNet, UNet_2
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader
from prepare_data import BioImageDataset
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
            scheduler.step()
            loop.set_postfix(loss=loss.item())
        length = len(train_loader.dataset)
        if best_acc < acc:
            best_acc = acc
            torch.save({
                'model_state_dict': net.state_dict(),
                'epoch': epoch,
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'weights/best_ckp.pth')
        print(f'Done epoch {epoch+1}/{num_epochs}: loss = {training_loss/length}, acc = {acc/len(train_loader)}')

def inference(image):
    model = UNet(1, 2)
    if os.path.exists('weights/best_ckp.pth'):
        ckp = torch.load('weights/best_ckp.pth')
        model.load_state_dict(ckp['model_state_dict'])
    threshold = 0.5

    model.eval()
    with torch.no_grad():
        image = Image.open(image)
        image = transforms.ToTensor()(image)
        output = model(image.unsqueeze(0))
        output = torch.sigmoid(output)
        output = output > threshold
        output = output.squeeze(0).squeeze(0)
        binary = torch.where(output, torch.tensor(255), torch.tensor(0))
        binary_image = Image.fromarray(binary.numpy().astype('uint8'))
        binary_image.save('a.png')
        return output


if __name__ == '__main__':
    net = UNet(1, 1)
    # net = net.cuda()
    print(summary(net, (1, 572, 572), device='cpu'))
    exit()
    # net_2 = UNet_2(1, 1)
    # net_2 = net_2.cuda()
    # print(summary(net_2, (1, 572, 572)))

    # epoch_start = 0
    # # if os.path.exists('weights/best_ckp.pth'):
    # #     print('Loading...')
    # #     checkpoint = torch.load('weights/best_ckp.pth')
    # #     net.load_state_dict(checkpoint['model_state_dict'])
    # #     epoch_start = checkpoint['epoch']
    # os.makedirs('weights', exist_ok=True)
    # criterion = nn.BCEWithLogitsLoss()
    #
    # optimizer = torch.optim.Adam(
    #     params=net.parameters(),
    #     lr=args.learning_rate,
    #     betas=[args.beta1, args.beta2],
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # train_image = BioImageDataset('data/membrane/train/aug')
    # train_loader = DataLoader(train_image, batch_size=args.batch_size, shuffle=True, drop_last=False)
    #
    # train(args.epochs, epoch_start)

    out = inference(r'D:\research about semantic segmentation\u-net\data\membrane\train\aug\image_14_3602105.png')
    mask = Image.open(r'D:\research about semantic segmentation\u-net\data\membrane\train\aug\mask_14_3602105.png')
    h, w = mask.size
    mask = transforms.ToTensor()(mask)
    print(sum(sum(sum(mask == out)))/(h * w))
