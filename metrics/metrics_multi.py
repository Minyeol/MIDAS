import torch
import piq
import pyiqa
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torchvision.transforms as T
import yaml
import glob
import argparse

transform = T.ToTensor()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, default='./dataset/steganodataset_v1/config.yaml')
    parser.add_argument('--num_secret_image', type=int, default=2)
    parser.add_argument("--path", type=str, default='./results/MIDAS/stego260/2/seed9000/')
    args = parser.parse_args()
    yaml_path = args.yaml_path
    with open(yaml_path, "r", encoding='utf-8') as f:
        yaml_list = yaml.safe_load(f)

    maniqa_stego = 0

    psnr_stego = 0
    psnr_correct = 0
    psnr_wrong = 0

    ssim_stego = 0
    ssim_correct = 0
    ssim_wrong = 0

    lpips_stego = 0
    lpips_correct = 0
    lpips_wrong = 0

    clip_stego = 0

    num_images = 0
    num_stego_images = 0
    num_correct = 0
    num_wrong = 0

    lpips_loss = piq.LPIPS(reduction='mean').eval()
    maniqa_model = pyiqa.create_metric('maniqa')

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")

    num_secret_images = args.num_secret_image
    for image_config in yaml_list:
        num_stego_images += 1
        image_path = args.path + str(num_stego_images) 
        stego_path = image_path + '_hide_pw_*.png'
        text_prompt = image_config['target_caption']

        image_stego = transform(Image.open(glob.glob(stego_path)[0])).unsqueeze(0).to(device)
        if num_secret_images == 2:
            image_stegos = [transform((T.ToPILImage()(image_stego[0, :, :,   0:256])).resize((512, 512))).unsqueeze(0).to(device), transform((T.ToPILImage()(image_stego[0, :, :,   256:512])).resize((512, 512))).unsqueeze(0).to(device)]
        elif num_secret_images == 4:
            image_stegos = [transform((T.ToPILImage()(image_stego[0, :, 0:256,   0:256])).resize((512, 512))).unsqueeze(0).to(device), 
                            transform((T.ToPILImage()(image_stego[0, :, 0:256,   256:512])).resize((512, 512))).unsqueeze(0).to(device),
                            transform((T.ToPILImage()(image_stego[0, :, 256:512,   0:256])).resize((512, 512))).unsqueeze(0).to(device),
                            transform((T.ToPILImage()(image_stego[0, :, 256:512,   256:512])).resize((512, 512))).unsqueeze(0).to(device)]
        
        elif num_secret_images == 8:
            image_stegos = [transform((T.ToPILImage()(image_stego[0, :, 0:256,   0:128])).resize((512, 512))).unsqueeze(0).to(device), 
                            transform((T.ToPILImage()(image_stego[0, :, 0:256,   128:256])).resize((512, 512))).unsqueeze(0).to(device),
                            transform((T.ToPILImage()(image_stego[0, :, 0:256,   256:384])).resize((512, 512))).unsqueeze(0).to(device),
                            transform((T.ToPILImage()(image_stego[0, :, 0:256,   384:512])).resize((512, 512))).unsqueeze(0).to(device),
                            transform((T.ToPILImage()(image_stego[0, :, 256:512,   0:128])).resize((512, 512))).unsqueeze(0).to(device), 
                            transform((T.ToPILImage()(image_stego[0, :, 256:512,   128:256])).resize((512, 512))).unsqueeze(0).to(device),
                            transform((T.ToPILImage()(image_stego[0, :, 256:512,   256:384])).resize((512, 512))).unsqueeze(0).to(device),
                            transform((T.ToPILImage()(image_stego[0, :, 256:512,   384:512])).resize((512, 512))).unsqueeze(0).to(device)]
        image_gts = []
        for i in range(num_secret_images):
            image_gt = transform(Image.open(image_path + '_' + str(i + 1) + '.png')).unsqueeze(0).to(device)
            psnr_stego += piq.psnr(image_gt, image_stegos[i], data_range=1.)
            ssim_stego += piq.ssim(image_gt, image_stegos[i], data_range=1.)
            lpips_stego += lpips_loss(image_gt, image_stegos[i])
            image_gts.append(image_gt)

        for i in range(num_secret_images):
            for j in range(num_secret_images):
                image_ij = transform(Image.open(glob.glob(image_path + '_rec_w_*_P' + str(i + 1) + '_' + str(j + 1) + '.png')[0])).unsqueeze(0).to(device)
                num_images += 1
                
                if i == j:
                    psnr_correct += piq.psnr(image_gts[i], image_ij, data_range=1.)
                    ssim_correct += piq.ssim(image_gts[i], image_ij, data_range=1.)
                    lpips_correct += lpips_loss(image_gts[i], image_ij)
                    num_correct += 1
                else:
                    psnr_wrong += piq.psnr(image_gts[j], image_ij, data_range=1.)
                    ssim_wrong += piq.ssim(image_gts[j], image_ij, data_range=1.)
                    lpips_wrong += lpips_loss(image_gts[j], image_ij)
                    num_wrong += 1
        maniqa_stego += maniqa_model(image_stego).item()
        image_uint8 = (image_stego.squeeze(0).clamp(0, 1) * 255).to(torch.uint8)
        clip_stego += clip_score_fn([image_uint8], [text_prompt]).detach()

    print('stego image')
    print(f'MANIQA : {maniqa_stego / num_stego_images:.4f}, CLIP Score : {clip_stego / num_stego_images:.4f}')
    print(f'PSNR : {psnr_stego / num_stego_images / num_secret_images:.4f}, SSIM : {ssim_stego / num_stego_images / num_secret_images:.4f}, LPIPS : {lpips_stego / num_stego_images / num_secret_images:.4f}')

    print('correct')
    print(f'PSNR : {psnr_correct / num_correct:.4f}, SSIM : {ssim_correct / num_correct:.4f}, LPIPS : {lpips_correct / num_correct:.4f}')

    print('wrong')
    print(f'PSNR : {psnr_wrong / num_wrong:.4f}, SSIM : {ssim_wrong / num_wrong:.4f}, LPIPS : {lpips_wrong / num_wrong:.4f}')
