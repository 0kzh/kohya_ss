import json
import math
import os
import pathlib
from library.common_gui import (
    save_inference_file,
    run_cmd_advanced_training,
    run_cmd_training,
    check_if_model_exist,
)
from library.sampler_gui import run_cmd_sample

def read_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    return config

config_file_path = "train_lora_config.json"
config = read_config(config_file_path)

def train_model(config):

    # load config
    print_only = False
    stop_text_encoder_training_pct = 0
    pretrained_model_name_or_path = config['pretrained_model_name_or_path']
    train_data_dir = config['train_data_dir']
    reg_data_dir = config['reg_data_dir']
    output_dir = config['output_dir']
    logging_dir = config['logging_dir']
    bucket_reso_steps = config['bucket_reso_steps']
    output_name = config['output_name']
    save_model_as = config['save_model_as']
    optimizer = config['optimizer']
    train_batch_size = config['train_batch_size']
    epoch = config['epoch']
    num_cpu_threads_per_process = config['num_cpu_threads_per_process']
    v2 = config['v2']
    v_parameterization = config['v_parameterization']
    enable_bucket = config['enable_bucket']
    no_token_padding = config['no_token_padding']
    weighted_captions = config['weighted_captions']
    max_resolution = config['max_resolution']
    network_alpha = config['network_alpha']
    training_comment = config['training_comment']
    prior_loss_weight = config['prior_loss_weight']
    LoRA_type = config['LoRA_type']
    conv_dim = config['conv_dim']
    conv_alpha = config['conv_alpha']
    network_dim = config['network_dim']
    lora_network_weights = config['lora_network_weights']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    lr_scheduler_num_cycles = config['lr_scheduler_num_cycles']
    lr_scheduler_power = config['lr_scheduler_power']
    learning_rate = config['learning_rate']
    lr_scheduler = config['lr_scheduler']
    save_every_n_epochs = config['save_every_n_epochs']
    mixed_precision = config['mixed_precision']
    save_precision = config['save_precision']
    seed = config['seed']
    caption_extension = config['caption_extension']
    cache_latents = config['cache_latents']
    optimizer_args = config['optimizer_args']
    max_train_epochs = config['max_train_epochs']
    max_train_epochs = config['max_train_epochs']
    max_data_loader_n_workers = config['max_data_loader_n_workers']
    max_token_length = config['max_token_length']
    resume = config['resume']
    save_state = config['save_state']
    mem_eff_attn = config['mem_eff_attn']
    clip_skip = config['clip_skip']
    flip_aug = config['flip_aug']
    color_aug = config['color_aug']
    shuffle_caption = config['shuffle_caption']
    gradient_checkpointing = config['gradient_checkpointing']
    full_fp16 = config['full_fp16']
    xformers = config['xformers']
    # use_8bit_adam = config['use_8bit_adam']  # Uncomment this line if 'use_8bit_adam' is present in the config
    keep_tokens = config['keep_tokens']
    persistent_data_loader_workers = config['persistent_data_loader_workers']
    bucket_no_upscale = config['bucket_no_upscale']
    random_crop = config['random_crop']
    caption_dropout_every_n_epochs = config['caption_dropout_every_n_epochs']
    caption_dropout_rate = config['caption_dropout_rate']
    noise_offset = config['noise_offset']
    additional_parameters = config['additional_parameters']
    vae_batch_size = config['vae_batch_size']
    min_snr_gamma = config['min_snr_gamma']
    sample_every_n_steps = config['sample_every_n_steps']
    sample_every_n_epochs = config['sample_every_n_epochs']
    sample_sampler = config['sample_sampler']
    sample_prompts = config['sample_prompts']
    text_encoder_lr = config['text_encoder_lr']
    unet_lr = config['unet_lr']
    lr_warmup = config['lr_warmup']

    if pretrained_model_name_or_path == '':
        raise Exception('Source model information is missing')

    if train_data_dir == '':
        raise Exception('Image folder path is missing')

    if not os.path.exists(train_data_dir):
        raise Exception('Image folder does not exist')

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            raise Exception('Regularisation folder does not exist')

    if output_dir == '':
        raise Exception('Output folder path is missing')

    if int(bucket_reso_steps) < 1:
        raise Exception('Bucket resolution steps must be greater than 0')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        print(
            'Output "stop text encoder training" is not yet supported. Ignoring'
        )
        stop_text_encoder_training_pct = 0

    if check_if_model_exist(output_name, output_dir, save_model_as):
        return
    
    if optimizer == 'Adafactor' and lr_warmup != '0':
        print("Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.", title="Warning")
        lr_warmup = '0'

    # If string is empty set string to 0.
    if text_encoder_lr == '':
        text_encoder_lr = 0
    if unet_lr == '':
        unet_lr = 0

    # if (float(text_encoder_lr) == 0) and (float(unet_lr) == 0):
    #     msgbox(
    #         'At least one Learning Rate value for "Text encoder" or "Unet" need to be provided'
    #     )
    #     return

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        # Extract the number of repeats from the folder name
        repeats = int(folder.split('_')[0])

        # Count the number of images in the folder
        num_images = len(
            [
                f
                for f, lower_f in (
                    (file, file.lower())
                    for file in os.listdir(
                        os.path.join(train_data_dir, folder)
                    )
                )
                if lower_f.endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]
        )

        print(f'Folder {folder}: {num_images} images found')

        # Calculate the total number of steps for this folder
        steps = repeats * num_images

        # Print the result
        print(f'Folder {folder}: {steps} steps')

        total_steps += steps

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(
            float(total_steps)
            / int(train_batch_size)
            * int(epoch)
            # * int(reg_factor)
        )
    )
    print(f'max_train_steps = {max_train_steps}')

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    print(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    print(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "train_network.py"'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += ' --enable_bucket'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    if weighted_captions:
        run_cmd += ' --weighted_captions'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution={max_resolution}'
    run_cmd += f' --output_dir="{output_dir}"'
    run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --network_alpha="{network_alpha}"'
    if not training_comment == '':
        run_cmd += f' --training_comment="{training_comment}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'
    if LoRA_type == 'LoCon' or LoRA_type == 'LyCORIS/LoCon':
        try:
            import lycoris
        except ModuleNotFoundError:
            print(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "algo=lora"'
    if LoRA_type == 'LyCORIS/LoHa':
        try:
            import lycoris
        except ModuleNotFoundError:
            print(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "algo=loha"'
    
        
    if LoRA_type in ['Kohya LoCon', 'Standard']:
        kohya_lora_var_list = ['down_lr_weight', 'mid_lr_weight', 'up_lr_weight', 'block_lr_zero_threshold', 'block_dims', 'block_alphas', 'conv_dims', 'conv_alphas']
        
        run_cmd += f' --network_module=networks.lora'
        kohya_lora_vars = {key: value for key, value in vars().items() if key in kohya_lora_var_list and value}

        network_args = ''
        if LoRA_type == 'Kohya LoCon':
            network_args += f' "conv_dim={conv_dim}" "conv_alpha={conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'
            
    if LoRA_type in ['Kohya DyLoRA']:
        kohya_lora_var_list = ['conv_dim', 'conv_alpha', 'down_lr_weight', 'mid_lr_weight', 'up_lr_weight', 'block_lr_zero_threshold', 'block_dims', 'block_alphas', 'conv_dims', 'conv_alphas', 'unit']
        
        run_cmd += f' --network_module=networks.dylora'
        kohya_lora_vars = {key: value for key, value in vars().items() if key in kohya_lora_var_list and value}

        network_args = ''

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if not (float(text_encoder_lr) == 0) or not (float(unet_lr) == 0):
        if not (float(text_encoder_lr) == 0) and not (float(unet_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --unet_lr={unet_lr}'
        elif not (float(text_encoder_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --network_train_text_encoder_only'
        else:
            run_cmd += f' --unet_lr={unet_lr}'
            run_cmd += f' --network_train_unet_only'
    else:
        if float(text_encoder_lr) == 0:
            raise ValueError('Please input text encoder learning rate value.')

    run_cmd += f' --network_dim={network_dim}'

    if not lora_network_weights == '':
        run_cmd += f' --network_weights="{lora_network_weights}"'
    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

    run_cmd += run_cmd_training(
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_epochs=save_every_n_epochs,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        seed=seed,
        caption_extension=caption_extension,
        cache_latents=cache_latents,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
    )

    run_cmd += run_cmd_advanced_training(
        max_train_epochs=max_train_epochs,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_token_length=max_token_length,
        resume=resume,
        save_state=save_state,
        mem_eff_attn=mem_eff_attn,
        clip_skip=clip_skip,
        flip_aug=flip_aug,
        color_aug=color_aug,
        shuffle_caption=shuffle_caption,
        gradient_checkpointing=gradient_checkpointing,
        full_fp16=full_fp16,
        xformers=xformers,
        # use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        noise_offset=noise_offset,
        additional_parameters=additional_parameters,
        vae_batch_size=vae_batch_size,
        min_snr_gamma=min_snr_gamma,
    )

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )
    
    # if not down_lr_weight == '':
    #     run_cmd += f' --down_lr_weight="{down_lr_weight}"'
    # if not mid_lr_weight == '':
    #     run_cmd += f' --mid_lr_weight="{mid_lr_weight}"'
    # if not up_lr_weight == '':
    #     run_cmd += f' --up_lr_weight="{up_lr_weight}"'
    # if not block_lr_zero_threshold == '':
    #     run_cmd += f' --block_lr_zero_threshold="{block_lr_zero_threshold}"'
    # if not block_dims == '':
    #     run_cmd += f' --block_dims="{block_dims}"'
    # if not block_alphas == '':
    #     run_cmd += f' --block_alphas="{block_alphas}"'
    # if not conv_dims == '':
    #     run_cmd += f' --conv_dims="{conv_dims}"'
    # if not conv_alphas == '':
    #     run_cmd += f' --conv_alphas="{conv_alphas}"'
        



    if print_only:
        print(
            '\033[93m\nHere is the trainer command as a reference. It will not be executed:\033[0m\n'
        )
        print('\033[96m' + run_cmd + '\033[0m\n')
    else:
        print(run_cmd)
        # Run the command
        if os.name == 'posix':
            os.system(run_cmd)
        else:
            subprocess.run(run_cmd)

        # check if output_dir/last is a folder... therefore it is a diffuser model
        last_dir = pathlib.Path(f'{output_dir}/{output_name}')

        if not last_dir.is_dir():
            # Copy inference model for v2 if required
            save_inference_file(
                output_dir, v2, v_parameterization, output_name
            )

def check_if_model_exist(output_name, output_dir, save_model_as):
    if save_model_as in ['diffusers', 'diffusers_safetendors']:
        ckpt_folder = os.path.join(output_dir, output_name)
        if os.path.isdir(ckpt_folder):
            msg = f'A diffuser model with the same name {ckpt_folder} already exists. Do you want to overwrite it? (yes/no) '
            user_input = input(msg).lower()
            if user_input != 'yes':
                print(
                    'Aborting training due to existing model with same name...'
                )
                return True
    elif save_model_as in ['ckpt', 'safetensors']:
        ckpt_file = os.path.join(output_dir, output_name + '.' + save_model_as)
        if os.path.isfile(ckpt_file):
            msg = f'A model with the same file name {ckpt_file} already exists. Do you want to overwrite it? (yes/no) '
            user_input = input(msg).lower()
            if user_input != 'yes':
                print(
                    'Aborting training due to existing model with same name...'
                )
                return True
    else:
        print(
            'Can\'t verify if existing model exist when save model is set a "same as source model", continuing to train model...'
        )
        return False

    return False

if __name__ == '__main__':
    train_model(config)