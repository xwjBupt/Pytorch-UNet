import argparse
import logging
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from pathlib import Path
from utils.data_loading import BasicDataset
from unet import UNet
from tqdm import tqdm


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):

    img = full_img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)[0]
        full_mask = probs.cpu().squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='Best_model.pth',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input_dir', '-i', default='/home/xwj/Xraw/DSA/FPDSA-Split/test/images/',
                        help='Filenames of input images')
    parser.add_argument('--save_dir', default='save_dir',
                        help='path to save predicted images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def mask_to_image(mask: np.ndarray):
    if mask.ndim <= 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    net = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.load_state_dict(torch.load(
        args.model, map_location=torch.device('cpu')))
    net.to(device=device)
    net.eval()
    logging.info('Model loaded!')
    for i, filename in tqdm(enumerate(glob.glob(args.input_dir+'/*.jpg'))):
        logging.info(f'\nPredicting image {filename} ...')
        img_name = os.path.basename(filename)
        img = cv2.imread(filename, 0)
        img = (img-img.mean())/img.std()
        img = torch.as_tensor(img.copy()).unsqueeze(
            0).unsqueeze(0).float().contiguous()
        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           device=device)
        out_filename = args.save_dir + '/'+img_name.replace('.jpg', '#pre.png')
        result = mask_to_image(mask)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')
