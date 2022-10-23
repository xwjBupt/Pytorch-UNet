import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset, DSA_VESSL
from utils.dice_score import dice_loss, BCEWithLogitsLoss2d
from evaluate import evaluate
from unet import UNet
import os


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_inter=1
              ):
    # 1. Create dataset
    train_dataset = DSA_VESSL(state='train')
    val_dataset = DSA_VESSL(state='val')
    max_value = -1
    max_epoch = -1
    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False,
                            drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net-4TH', resume='allow')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
                                  ))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Device:          {device.type}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2)  # goal: maximize Dice score
    criterion = BCEWithLogitsLoss2d()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_description('loss (batch) {} \n'.format(loss.item()))

            val_inter = 1
            if epoch % val_inter == 0:
                val_score = evaluate(net, val_loader, device)
                scheduler.step(val_score)
                logging.info(
                    'Validation Dice score: {} at epoch {}'.format(val_score, epoch))
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                })
                if val_score > max_value:
                    if hasattr(net, 'module'):
                        net_dict = net.module.state_dict()
                    else:
                        net_dict = net.state_dict()
                    torch.save(net_dict, 'Best_model.pth')
                    logging.info(
                        f'Checkpoint {epoch} saved as Best_model')
                    max_value = val_score
                    max_epoch = epoch


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int,
                        default=1, help='Number of classes')
    parser.add_argument('--gpu_ids', type=str,
                        default='1,2', help='ID of using GPUs')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    if len(args.gpu_ids) > 1:
        net = torch.nn.DataParallel(net)
    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
