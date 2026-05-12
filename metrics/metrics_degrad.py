import torch
import piq
import pyiqa
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torchvision.transforms as T
import yaml
import os 
import glob
import argparse

transform = T.ToTensor()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, default='./dataset/steganodataset_v1/config.yaml')
    parser.add_argument('--num_secret_image', type=int, default=2)
    parser.add_argument("--image_root", type=str, default='./results/MIDAS/stego260/2/seed9000/')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    with open(yaml_path, "r", encoding='utf-8') as f:
        yaml_list = yaml.safe_load(f)

    psnr_wo_degrade = 0
    psnr_jpg70 = 0
    psnr_Gnoise5 = 0
    psnr_Gblur2 = 0

    ssim_wo_degrade = 0
    ssim_jpg70 = 0
    ssim_Gnoise5 = 0
    ssim_Gblur2 = 0

    lpips_wo_degrade = 0
    lpips_jpg70 = 0
    lpips_Gnoise5 = 0
    lpips_Gblur2 = 0
    
    num_stego_images = 0
    num_images = 0
    num_secret_images = args.num_secret_image
    lpips_loss = piq.LPIPS(reduction='mean')

    for image_config in yaml_list:
        num_stego_images += 1
        image_path = args.image_root + str(num_stego_images)

        for i in range(num_secret_images):
            num_images += 1
            image_gt = transform(Image.open(image_path + '_' + str(i + 1) + '.png')).unsqueeze(0).to(device)
            image_wo_degrade = transform(Image.open(glob.glob(image_path + '_' + str(i + 1)  + '_rec_wo_degrad.png')[0])).unsqueeze(0).to(device)
            image_jpg70 = transform(Image.open(glob.glob(image_path + '_' + str(i + 1) + '_rec_JPG_70.png')[0])).unsqueeze(0).to(device)
            image_Gnoise5 = transform(Image.open(glob.glob(image_path + '_' + str(i + 1) + '_rec_Gnoise_5.png')[0])).unsqueeze(0).to(device)
            image_Gblur2 = transform(Image.open(glob.glob(image_path + '_' + str(i + 1) + '_rec_Gblur_2.png')[0])).unsqueeze(0).to(device)
            psnr_wo_degrade += piq.psnr(image_gt, image_wo_degrade, data_range=1.)
            psnr_jpg70 += piq.psnr(image_gt, image_jpg70, data_range=1.)
            psnr_Gnoise5 += piq.psnr(image_gt, image_Gnoise5, data_range=1.)
            psnr_Gblur2 += piq.psnr(image_gt, image_Gblur2, data_range=1.)

            ssim_wo_degrade += piq.ssim(image_gt, image_wo_degrade, data_range=1.)
            ssim_jpg70 += piq.ssim(image_gt, image_jpg70, data_range=1.)
            ssim_Gnoise5 += piq.ssim(image_gt, image_Gnoise5, data_range=1.)
            ssim_Gblur2 += piq.ssim(image_gt, image_Gblur2, data_range=1.)
            lpips_wo_degrade += lpips_loss(image_gt, image_wo_degrade)
            lpips_jpg70 += lpips_loss(image_gt, image_jpg70)
            lpips_Gnoise5 += lpips_loss(image_gt, image_Gnoise5)
            lpips_Gblur2 += lpips_loss(image_gt, image_Gblur2)

    print('without degrade')
    print(f'PSNR : {psnr_wo_degrade / num_images:.4f}, SSIM : {ssim_wo_degrade / num_images:.4f}, LPIPS : {lpips_wo_degrade / num_images:.4f}')

    print('JPG Q=70')
    print(f'PSNR : {psnr_jpg70 / num_images:.4f}, SSIM : {ssim_jpg70 / num_images:.4f}, LPIPS : {lpips_jpg70 / num_images:.4f}')

    print('Gaussian Noise 5')
    print(f'PSNR : {psnr_Gnoise5 / num_images:.4f}, SSIM : {ssim_Gnoise5 / num_images:.4f}, LPIPS : {lpips_Gnoise5 / num_images:.4f}')

    print('Gaussian Blur 2')
    print(f'PSNR : {psnr_Gblur2 / num_images:.4f}, SSIM : {ssim_Gblur2 / num_images:.4f}, LPIPS : {lpips_Gblur2 / num_images:.4f}')