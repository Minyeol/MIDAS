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
    parser.add_argument("--path", type=str, default='results/MIDAS/stego260/1/seed9000/')
    args = parser.parse_args()
    yaml_path = args.yaml_path
    with open(yaml_path, "r", encoding='utf-8') as f:
        yaml_list = yaml.safe_load(f)

    psnr_stego = 0
    psnr_recover = 0
    psnr_wrong = 0

    ssim_stego = 0
    ssim_recover = 0
    ssim_wrong = 0

    lpips_stego = 0
    lpips_recover = 0
    lpips_wrong = 0

    clip_stego = 0

    num_images = 0
    lpips_loss = piq.LPIPS(reduction='mean')

    maniqa_stego = 0
    maniqa_model = pyiqa.create_metric('maniqa')

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")

    for image_config in yaml_list:
        num_images += 1
        image_path = args.path + os.path.basename(image_config['image_path'])
        text_prompt = image_config['target_caption']
        if image_path.endswith('.png') or image_path.endswith('.jpg'):
            image_gt = transform(Image.open(image_path[:-4] + '.png')).unsqueeze(0).to(device)
            image_stego = transform(Image.open(glob.glob(image_path[:-4] + '_hide_pw_*.png')[0])).unsqueeze(0).to(device)
            image_recover = transform(Image.open(glob.glob(image_path[:-4] + '_rec_w_*.png')[0])).unsqueeze(0).to(device)
            image_wrong = transform(Image.open(glob.glob(image_path[:-4] + '_rec_wo_*_w.png')[0])).unsqueeze(0).to(device)
        elif image_path.endswith('.jpeg'):
            image_gt = transform(Image.open(image_path[:-5] + '.png')).unsqueeze(0).to(device)
            image_stego = transform(Image.open(glob.glob(image_path[:-5] + '_hide_pw_*.png')[0])).unsqueeze(0).to(device)
            image_recover = transform(Image.open(glob.glob(image_path[:-5] + '_rec_w_*.png')[0])).unsqueeze(0).to(device)
            image_wrong = transform(Image.open(glob.glob(image_path[:-5] + '_rec_wo_*_w.png')[0])).unsqueeze(0).to(device)
        psnr_stego += piq.psnr(image_gt, image_stego, data_range=1.)
        psnr_recover += piq.psnr(image_gt, image_recover, data_range=1.)
        psnr_wrong += piq.psnr(image_gt, image_wrong, data_range=1.)

        ssim_stego += piq.ssim(image_gt, image_stego, data_range=1.)
        ssim_recover += piq.ssim(image_gt, image_recover, data_range=1.)
        ssim_wrong += piq.ssim(image_gt, image_wrong, data_range=1.)

        lpips_stego += lpips_loss(image_gt, image_stego)
        lpips_recover += lpips_loss(image_gt, image_recover)
        lpips_wrong += lpips_loss(image_gt, image_wrong)

        maniqa_stego += maniqa_model(image_stego)
        image_uint8 = (image_stego.squeeze(0).clamp(0, 1) * 255).to(torch.uint8)
        clip_stego += clip_score_fn([image_uint8], [text_prompt]).detach()

    print('cover image')
    print(f'MANIQA : {maniqa_stego.item() / num_images:.4f}, CLIP Score : {clip_stego / num_images:.4f}')
    print(f'PSNR : {psnr_stego / num_images:.4f}, SSIM : {ssim_stego / num_images:.4f}, LPIPS : {lpips_stego / num_images:.4f}')

    print('recovered image')
    print(f'PSNR : {psnr_recover / num_images:.4f}, SSIM : {ssim_recover / num_images:.4f}, LPIPS : {lpips_recover / num_images:.4f}')

    print('wrong password')
    print(f'PSNR : {psnr_wrong / num_images:.4f}, SSIM : {ssim_wrong / num_images:.4f}, LPIPS : {lpips_wrong / num_images:.4f}')
