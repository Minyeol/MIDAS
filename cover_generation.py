import torch
import yaml
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
import os
import random
from PIL import Image

from scheduling import EDICTScheduler
from ip_adapter import IPAdapterPlus

class ODESolve:
    def __init__(self, args, EXT_SCALE=0., p=0.93):
        self.args = args
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        ref_key = self.args.ref_key
        ref_stable = None
        # use another model in the reverse diffusion of hiding stage
        # and the forward diffusion of the recovery stage
        self.ref_model = not args.single_model
        ldm_stable = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",    # majicmix-fantasy
                                                            safety_checker=None,
                                                            torch_dtype=torch.float16).to(self.device)
        if self.ref_model:
            if ref_key[0] == ".":
                ref_stable = StableDiffusionPipeline.from_single_file(ref_key,
                                                                        safety_checker=None,
                                                                        torch_dtype=torch.float16).to(self.device)
            else:
                ref_stable = StableDiffusionPipeline.from_pretrained(ref_key,
                                                                        safety_checker=None,
                                                                        torch_dtype=torch.float16).to(self.device)
            ref_stable.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                                        algorithm_type="sde-dpmsolver++")

        try:
            ldm_stable.disable_xformers_memory_efficient_attention()
            if ref_stable is not None:
                ref_stable.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        self.model = ldm_stable
        self.model_ref = ldm_stable if args.single_model else ref_stable
        self.guidance_scale = self.args.guidance_scale
        self.ext_scale = EXT_SCALE
        self.edict_scheduler = EDICTScheduler(p=p, ext_scale=EXT_SCALE)
        self.num_steps = args.num_steps
        self.tokenizer = self.model.tokenizer
        self.edict_scheduler.set_timesteps(self.num_steps, device=self.model.device)
        self.prompt = None
        self.context = None
        # EDICT: "In practice, we alternate the order in which the x and y series are calculated
        # at each step in order to symmetrize the process with respect to both sequences."
        self.leapfrog_steps = True
        # diffusion steps within [0, self.strength * T]
        self.strength = args.edit_strength
        base_path = os.path.dirname(os.path.abspath(__file__))
        ip_ckpt = os.path.join(base_path, "./models/ip-adapter-plus_sd15.bin")
        if self.ref_model:
            ip_model_load = self.model_ref
        else:
            ip_model_load = self.model
        self.ip_model = IPAdapterPlus(ip_model_load, os.path.join(base_path, "./models/"), ip_ckpt, self.device, num_tokens=16)
        self.ip_edit_strength = args.ip_edit_strength
        self.ip_edit_start = args.ip_edit_start
        self.ip_model.set_scale(0.)
        self.ip_scale = args.ip_scale

    @torch.no_grad()
    def latent2image(self, latents, seed, return_type='np', use_sc=False, ref_vae=False):
        if isinstance(latents, list):
            latents = latents[0]
        latents = 1 / 0.18215 * latents.detach()
        vae = self.model_ref.vae if ref_vae else self.model.vae
        if use_sc:
            image = self.vae_sc.decode(latents, generator=torch.Generator(device=latents.device).manual_seed(seed))[
                'sample']
        else:
            image = vae.decode(latents, generator=torch.Generator(device=latents.device).manual_seed(seed))['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image, use_sc=False, ref_vae=False):
        vae = self.model_ref.vae if ref_vae else self.model.vae
        with torch.no_grad():
            if type(image) is Image.Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device).to(torch.float16)
                if use_sc:
                    latents = self.vae_sc.encode(image)['latent_dist'].mean
                else:
                    latents = vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def edict_noise(self, latent: torch.Tensor, ref_unet=False, full_inversion=False, control_image=None):
        coupled_latents = [latent.clone(), latent.clone()]
        latent_list = []
        noise_pred_list = []
        h_space_list = []

        if full_inversion:
            t_limit = 0
        else:
            t_limit = self.num_steps - int(self.num_steps * self.strength)
        timesteps = self.edict_scheduler.timesteps[t_limit:].flip(0)
        model = self.model_ref if ref_unet else self.model
        unet = model.unet
        if control_image is not None:
            controlnet = model.controlnet
            control_image = model.prepare_image(
                image=control_image,
                width=512,
                height=512,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            coupled_latents = self.edict_scheduler.noise_mixing_layer(
                x=coupled_latents[0], y=coupled_latents[1]
            )
            # t = self.edict_scheduler.timesteps[len(self.edict_scheduler.timesteps) - i - 1]
            # j - model_input index, k - base index
            for j in range(2):
                k = j ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 0:
                        k, j = j, k

                model_input = coupled_latents[j]
                base = coupled_latents[k]

                model_input = model_input.repeat(2,1,1,1)

                if control_image is None:
                    unet_output = unet(model_input, t, self.context)
                else:
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        model_input,
                        t,
                        encoder_hidden_states=self.context,
                        controlnet_cond=control_image,
                        return_dict=False,
                        conditioning_scale=1.0,
                    )

                    unet_output = unet(
                        model_input, t, encoder_hidden_states=self.context,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )

                noise_pred_uncond, noise_prediction_text = unet_output["sample"].chunk(2)

                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)

                noise_pred_list.append(noise_pred)  # [ε_{x^{inter}_1}, ε_{y_1}, ..., ε_{x^{inter}_T}, ε_{y_T}]
                base, model_input = self.edict_scheduler.noise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )
                latent_list.append(model_input)  # [y_1, x_1, ..., y_T, x_T]
                coupled_latents[k] = model_input

        return coupled_latents, latent_list, noise_pred_list, h_space_list

    @torch.no_grad()
    def edict_denoise(self, latent_pair: list, ref_unet=False, full_inversion=False, control_image=None):
        latent_list = []
        noise_pred_list = []
        h_space_list = []

        if full_inversion:
            t_limit = 0
        else:
            t_limit = self.num_steps - int(self.num_steps * self.strength)
        timesteps = self.edict_scheduler.timesteps[t_limit:]
        model = self.model_ref if ref_unet else self.model
        unet = model.unet
        if control_image is not None:
            controlnet = model.controlnet
            control_image = model.prepare_image(
                image=control_image,
                width=512,
                height=512,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # t = self.edict_scheduler.timesteps[i]
            # j - model_input index, k - base index
            for k in range(2):
                j = k ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 1:
                        k, j = j, k
                    # if random_array[i]==0:k, j = j, k

                model_input = latent_pair[j]
                base = latent_pair[k]

                model_input = model_input.repeat(2, 1, 1, 1)

                if control_image is None:
                    unet_output = unet(model_input, t, self.context)
                else:
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        model_input,
                        t,
                        encoder_hidden_states=self.context,
                        controlnet_cond=control_image,
                        return_dict=False,
                        conditioning_scale=1.0,
                    )

                    unet_output = unet(
                        model_input, t, encoder_hidden_states=self.context,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )

                noise_pred_uncond, noise_prediction_text = unet_output["sample"].chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)

                noise_pred_list.append(noise_pred)  # [ε_{y_T}, ε_{x^{inter}_T}, ..., ε_{y_1}, ε_{x^{inter}_1}]

                base, model_input = self.edict_scheduler.denoise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )

                latent_pair[k] = model_input

            latent_pair = self.edict_scheduler.denoise_mixing_layer(
                x=latent_pair[0], y=latent_pair[1]
            )

            # [x_{T-1}, y_{T-1}, ..., x_0, y_0]
            latent_list.append(latent_pair[0])
            latent_list.append(latent_pair[1])
        return latent_pair, latent_list, noise_pred_list, h_space_list

    @torch.no_grad()
    def edict_denoise_edit_with_ip_adapter(self, latent_pair: list, ip_embeds: torch.Tensor = None):
        if ip_embeds is None:
            return self.edict_denoise(latent_pair)

        target_embeddings_ip = self.encode_merge(self.context, ip_embeds)

        new_latent_list = []

        t_limit = self.num_steps - int(self.num_steps * self.strength)
        timesteps = self.edict_scheduler.timesteps[t_limit:]

        unet = self.model_ref.unet if self.ref_model else self.model.unet

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # j - model_input index, k - base index
            for k in range(2):
                j = k ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 1:
                        k, j = j, k

                model_input = latent_pair[j]
                base = latent_pair[k]

                model_input = model_input.repeat(2, 1, 1, 1)


                if i >= int(len(timesteps) * self.ip_edit_start) and i < int(len(timesteps) * (self.ip_edit_strength + self.ip_edit_start)):
                    self.ip_model.set_scale(self.ip_scale)
                    unet_output = unet(model_input, t, target_embeddings_ip)
                    noise_pred_uncond, noise_prediction_text = unet_output["sample"].chunk(2)
                else:
                    self.ip_model.set_scale(0)
                    unet_output = unet(model_input, t, self.context)
                    noise_pred_uncond, noise_prediction_text = unet_output["sample"].chunk(2)

                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)
                base, model_input = self.edict_scheduler.denoise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t
                )

                latent_pair[k] = model_input

            latent_pair = self.edict_scheduler.denoise_mixing_layer(
                x=latent_pair[0], y=latent_pair[1]
            )

            # [x_{T-1}, y_{T-1}, ..., x_0, y_0]
            new_latent_list.append(latent_pair[0])
            new_latent_list.append(latent_pair[1])
        return latent_pair, new_latent_list

    @torch.no_grad()
    def edict_denoise_edit(self, latent_pair: list, latent_list: list = None):
        if latent_list is None:
            return self.edict_denoise(latent_pair)

        uncond_embeddings, cond_embeddings = self.context.chunk(2)

        new_latent_list = []

        t_limit = self.num_steps - int(self.num_steps * self.strength)
        timesteps = self.edict_scheduler.timesteps[t_limit:]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # Edit
            if len(latent_list) > 0:
                # latent_list = [y_1, x_1, ..., y_T, x_T]
                delta_x = latent_list.pop()
                delta_y = latent_list.pop()
                latent_pair[0] = delta_x * self.ext_scale + latent_pair[0] * (1 - self.ext_scale)
                latent_pair[1] = delta_y * self.ext_scale + latent_pair[1] * (1 - self.ext_scale)

            # j - model_input index, k - base index
            for k in range(2):
                j = k ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 1:
                        k, j = j, k
                    # if random_array[i]==0:k, j = j, k

                model_input = latent_pair[j]
                base = latent_pair[k]

                noise_pred_uncond = self.model.unet(model_input, t, uncond_embeddings)["sample"]
                noise_prediction_text = self.model.unet(model_input, t, cond_embeddings)["sample"]
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)

                base, model_input = self.edict_scheduler.denoise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )

                latent_pair[k] = model_input

            latent_pair = self.edict_scheduler.denoise_mixing_layer(
                x=latent_pair[0], y=latent_pair[1]
            )

            # [x_{T-1}, y_{T-1}, ..., x_0, y_0]
            new_latent_list.append(latent_pair[0])
            new_latent_list.append(latent_pair[1])
        return latent_pair, new_latent_list

    @torch.no_grad()
    def edict_noise_rec_with_ip_adapter(self, latent: torch.Tensor, ip_embeds: torch.Tensor = None):
        if ip_embeds is None:
            return self.edict_noise(latent)

        coupled_latents = [latent.clone(), latent.clone()]

        target_embeddings_ip = self.encode_merge(self.context, ip_embeds)

        new_latent_list = []

        t_limit = self.num_steps - int(self.num_steps * self.strength)
        timesteps = self.edict_scheduler.timesteps[t_limit:].flip(0)
        unet = self.model_ref.unet if self.ref_model else self.model.unet

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            coupled_latents = self.edict_scheduler.noise_mixing_layer(
                x=coupled_latents[0], y=coupled_latents[1]
            )

            # j - model_input index, k - base index
            for j in range(2):
                k = j ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 0:
                        k, j = j, k

                model_input = coupled_latents[j]
                base = coupled_latents[k]

                model_input = model_input.repeat(2, 1, 1, 1)

                if i >= int(len(timesteps) * (1-self.ip_edit_strength - self.ip_edit_start)) and i < int(len(timesteps) * (1 - self.ip_edit_start)):
                    self.ip_model.set_scale(self.ip_scale)
                    unet_output = unet(model_input, t, target_embeddings_ip)
                    noise_pred_uncond, noise_prediction_text = unet_output["sample"].chunk(2)
                else:
                    self.ip_model.set_scale(0.)
                    unet_output = unet(model_input, t, self.context)
                    noise_pred_uncond, noise_prediction_text = unet_output["sample"].chunk(2)

                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)
                base, model_input = self.edict_scheduler.noise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t
                )

                new_latent_list.insert(0, model_input)
                coupled_latents[k] = model_input

        return coupled_latents, new_latent_list

    @torch.no_grad()
    def edict_noise_rec(self, latent: torch.Tensor, latent_list: list = None, ref_unet=False):
        if latent_list is None:
            return self.edict_noise(latent)

        coupled_latents = [latent.clone(), latent.clone()]

        # [y_1, x_1, ..., y_T, x_T] -> [x_T, y_T, ..., x_1, y_1]
        latent_list = latent_list[::-1]  # reverse
        new_latent_list = []

        uncond_embeddings, cond_embeddings = self.context.chunk(2)

        t_limit = self.num_steps - int(self.num_steps * self.strength)
        timesteps = self.edict_scheduler.timesteps[t_limit:].flip(0)
        unet = self.model_ref.unet if ref_unet else self.model.unet

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            coupled_latents = self.edict_scheduler.noise_mixing_layer(
                x=coupled_latents[0], y=coupled_latents[1]
            )

            # j - model_input index, k - base index
            for j in range(2):
                k = j ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 0:
                        k, j = j, k

                model_input = coupled_latents[j]
                base = coupled_latents[k]

                noise_pred_uncond = unet(model_input, t, uncond_embeddings)["sample"]
                noise_prediction_text = unet(model_input, t, cond_embeddings)["sample"]
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)

                base, model_input = self.edict_scheduler.noise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )
                new_latent_list.append(model_input)
                coupled_latents[k] = model_input

            # Rec
            if len(latent_list) > 0:
                # latent_list = [x_T, y_T, ..., x_1, y_1]
                delta_y = latent_list.pop()
                delta_x = latent_list.pop()
                coupled_latents[0] = (coupled_latents[0] - delta_x * self.ext_scale) / (1 - self.ext_scale)
                coupled_latents[1] = (coupled_latents[1] - delta_y * self.ext_scale) / (1 - self.ext_scale)

        return coupled_latents, new_latent_list

    def encode_merge(self, prompt_context, prompt_ip, use_text_prompt=True):
        if not use_text_prompt:
            prompt_context[1] = prompt_context[0]
        negative_prompt_embeds_1, prompt_embeds_1 = prompt_context.chunk(2)
        negative_prompt_embeds_2, prompt_embeds_2 = prompt_ip.chunk(2)
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=1)
        return torch.cat([negative_prompt_embeds, prompt_embeds])

    def edict_invert(self, prompt, start_latent, is_forward, require_list=False, ref_unet=False, full_inversion=False, control_image=None):
        self.init_prompt(prompt)
        if is_forward:
            latents, latent_list, noise_pred_list, h_space_list = self.edict_noise(start_latent, ref_unet=ref_unet, full_inversion=full_inversion,control_image=control_image)
        else:
            latents, latent_list, noise_pred_list, h_space_list = self.edict_denoise([x.clone() for x in start_latent], ref_unet=ref_unet, full_inversion=full_inversion,control_image=control_image)
        if require_list:
            return latents, latent_list, noise_pred_list, h_space_list
        else:
            return latents

    def edict_invert_edit(self, prompt, start_latent, is_forward, latent_list=None, ip_embeds=None):
        self.init_prompt(prompt)
        if is_forward:
            latents = self.edict_noise(start_latent)
        else:
            if ip_embeds is not None:
                latents, latent_list = self.edict_denoise_edit_with_ip_adapter(
                    [x.clone() for x in start_latent], ip_embeds.clone())
            else:
                latents, latent_list = self.edict_denoise_edit([x.clone() for x in start_latent], latent_list.copy())
        return latents, latent_list

    def edict_invert_rec(self, prompt, start_latent, is_forward, latent_list=None, ip_embeds=None):
        self.init_prompt(prompt)
        if is_forward:
            if ip_embeds is not None:
                latents, latent_list = self.edict_noise_rec_with_ip_adapter(start_latent,
                                                                            ip_embeds.clone())
            else:
                latents, latent_list = self.edict_noise_rec(start_latent, latent_list.copy())
        else:
            latents = self.edict_denoise([x.clone() for x in start_latent])
        return latents, latent_list

    def set_ip_param(self, ip_scale=0., ip_edit_strength=0.):
        self.ip_model.set_scale(ip_scale)
        self.ip_scale = ip_scale
        self.ip_edit_strength = ip_edit_strength


@torch.no_grad()
def cover_generation(args):
    ip_scale = args.ip_scale
    ip_edit_strength = args.ip_edit_strength

    with open(args.yaml_path, "r", encoding='utf-8') as f:
            yaml_list = yaml.safe_load(f)

    ode = None
    count = 0
    
    for image_config in yaml_list:
        prompt_2 = image_config["target_caption"]
        prompt_2_edict = prompt_2

        if args.rand_seed:  # use random seed for each sample
            args.seed = torch.randint(1,999999,(1,)).item()

        count += 1
        image_name = count
        
        ode = ODESolve(args)
        
        latent_noise = torch.randn(1, 4, 64, 64).to(torch.float16).to('cuda:0')
        ref_image = ode.model_ref(prompt_2 + args.additional_prompts,
                                  512, 512, num_inference_steps=args.num_steps,
                                  negative_prompt=args.negative_prompts,
                                  generator=torch.Generator(device="cuda:0").manual_seed(args.seed)).images[0]
        ode.set_ip_param(ip_scale=ip_scale, ip_edit_strength=ip_edit_strength)
        cond_ref_image_prompt_embeds, uncond_ref_image_prompt_embeds = ode.ip_model.get_image_embeds(ref_image)
        ref_image_prompt_embeds = torch.cat([uncond_ref_image_prompt_embeds, cond_ref_image_prompt_embeds])

        image_hide_latent, latent_list_image_hide = ode.edict_invert_edit(prompt_2_edict, [latent_noise.clone(), latent_noise.clone()], is_forward=False,
                                                                          ip_embeds=ref_image_prompt_embeds)
        image_hide = ode.latent2image(image_hide_latent[0], seed=args.seed, ref_vae=args.ref_model)
        cv2.imwrite("{:s}/{:d}_SD.png".format(args.save_path, image_name),
                    cv2.cvtColor(image_hide, cv2.COLOR_RGB2BGR))
        
def set_seed(seed):
    
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # train speed is slower after enabling this opts.

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pw', type=int, default=9000)
    parser.add_argument('--yaml_path', type=str, default='./dataset/steganodataset_v1/config.yaml', help='path to dataset config')
    parser.add_argument('--save_path', type=str, default='./results/MIDAS/stego260/cover/')
    parser.add_argument('--num_steps', type=int, default=50, help='sampling step of diffusion')

    parser.add_argument('--edit_strength', type=float, default=1, help='edit period of diffusion process')
    parser.add_argument('--ip_edit_strength', type=float, default=1., help='edit period of ip-adapter')
    parser.add_argument('--ip_edit_start', type=float, default=0., help='edit start of ip-adapter')
    parser.add_argument('--ip_scale', type=float, default=1.0, help='scale of ip-adapter')
    parser.add_argument('--guidance_scale', type=float, default=1, help='scale of cond')

    parser.add_argument('--single_model', default=False, action='store_true',
                        help="use single model for the whole hiding and recovery process")
    parser.add_argument('--rand_pw', default=True, action='store_true',
                        help="use different random private password for each sample")
    parser.add_argument('--additional_prompts', type=str, default=', dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD',
                        help="additional prompts to enhance the quality of encrypted image")
    parser.add_argument('--negative_prompts', type=str, default='(deformed iris, deformed pupils, semi-realistic, cgi, 3d, b&w, monochrome, render, sketch, cartoon, drawing, anime), cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck',# 'UnrealisticDream',
                        help="negative prompts")
    parser.add_argument('--ref_key', type=str, default='GraydientPlatformAPI/picx-real',
                        help="model key or path to the second model if it is used")
    args = parser.parse_args()
    assert args.ip_edit_strength + args.ip_edit_start <= 1

    args.seed = args.pw
    args.ref_model = not args.single_model
    args.rand_seed = args.rand_pw
    set_seed(args.seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    cover_generation(args)