from pathlib import Path
import torch
import math
import librosa
from subprocess import run
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # this is a workaround for matplotlib
from PIL import Image
import numpy as np
import random
# import fire
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import utils.bending as util
from utils.wrapper import StreamDiffusionWrapper



# set constants
FPS = 20
TXT2IMG = True
SAMPLING_RATE = 48000
IMAGE_STORAGE_PATH = Path("./image_outputs")
OUTPUT_VIDEO_PATH = Path("./video_outputs")
util.set_sampling_rate(SAMPLING_RATE)

DEBUG = False

bending_functions_range = {
    util.add_full: (0, 5),
    util.multiply: (0, 10),  # the min could be 1 or 0
    util.add_sparse: (),
    util.add_noise: (0, 3),
    util.subtract_full: (0, 5),
    util.threshold: (0, 4),
    util.soft_threshold: (0, 2),
    util.soft_threshold2: (0, 2),
    util.inversion: (0, 5),
    util.inversion2: (),
    util.log: (0, 10),
    util.power: (),
    util.rotate_z: (0, 2 * math.pi),
    util.rotate_x: (0, 2 * math.pi),
    util.rotate_y: (0, 2 * math.pi),
    util.rotate_y2: (0, 2 * math.pi),
    util.reflect: (),
    util.hadamard1: (),
    util.hadamard2: (),
    util.absolute: ()
}


def generate_video(audio_path, prompt_file_path, layer, bend_function, audio_feature, smoothing_fn, seed, width=512, height=512):
    if bend_function is not None:
        bend_function_name = bend_function.__name__
    else:
        bend_function_name = None
    audio_feature_name = audio_feature.__name__

    # initialize paths
    OUTPUT_VIDEO_PATH.mkdir(exist_ok=True)
    IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
    util.clear_dir(IMAGE_STORAGE_PATH)

    prompt = ""  # set this if NOT using txt2img
    negative_prompt = "low quality, text, words, letters"
    output = IMAGE_STORAGE_PATH
    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # model_id_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    # model_id_or_path = "C:\\Users\dzluk\\untitled-visualization-project\models\ldm\stable-diffusion-v1\model.ckpt"
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    lora_dict = None
    # width = 512
    # height = 512
    frame_buffer_size = 1  # batch size
    acceleration = "xformers"
    guidance_scale = 1.2
    do_classifier_free_guidance = True if guidance_scale > 1.0 else False
    t_index_list = [0, 16, 32, 45]  # I think that the number of layers equals the len of this list

    def txt2img(wrapper, prompt, noise, bending_fn):
        wrapper.prepare(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            bending_fn=bending_fn,
            input_noise=noise
        )

        count = len(list(output.iterdir()))
        output_images = wrapper()
        output_images.save(os.path.join(output, f"{count:05}.png"))


    def encoding2img(wrapper, encoding, noise, bending_fn):
        wrapper.prepare(
            prompt_encoding=encoding,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            bending_fn=bending_fn,
            input_noise=noise
        )
        count = len(list(output.iterdir()))
        output_images = wrapper()
        output_images.save(os.path.join(output, f"{count:05}.png"))


    def encode_prompt(wrapper, prompt):
        encoder_output = wrapper.stream.pipe.encode_prompt(
            prompt=prompt,
            device=wrapper.stream.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        prompt_embeds = encoder_output[0].repeat(wrapper.stream.batch_size, 1, 1)
        return prompt_embeds


    wrapper = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=t_index_list,  # the length of this list is the number of denoising steps
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="txt2img",
        use_denoising_batch=False,
        cfg_type="none",
        seed=seed,
        bending_fn=bend_function
    )


    if not TXT2IMG:
        # generate first frame using txt2img
        txt2img(wrapper, prompt, None, None)
        # create img2img stream
        wrapper = StreamDiffusionWrapper(
                model_id_or_path=model_id_or_path,
                lora_dict=lora_dict,
                t_index_list=[22, 32, 45],
                frame_buffer_size=1,  # this is batch size
                width=width,
                height=height,
                warmup=10,
                acceleration=acceleration,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="self",
                seed=seed,
        )


    # helper functions
    tic = time.time()

    # seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # read prompt file with format:
    # 0:00 : First prompt
    # 0:10 : Second prompt
    # etc...
    with open(prompt_file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        prompts = [line.split(":")[-1] for line in lines]
        times = [float(line.split(":")[0]) * 60 + float(line.split(":")[1]) for line in lines]
        prompt_times = [int(time * FPS) for time in times]
    num_prompts = len(prompts)

    # load input audio
    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)

    # calculate number of frames needed
    frame_length = 1. / FPS  # length of each frame in seconds
    num_frames = int(audio.size // (frame_length * SAMPLING_RATE))
    print(f">>> Generating {num_frames} frames")

    # Adding sin wave noise
    walk_length = 0.2  # set to 2 for 2pi walk
    noise = torch.empty((1, 4, wrapper.stream.latent_height, wrapper.stream.latent_width), dtype=torch.float64)
    walk_noise_x = torch.normal(mean=0, std=1, size=noise.shape, dtype=torch.float64)
    walk_noise_y = torch.normal(mean=0, std=1, size=noise.shape, dtype=torch.float64)
    walk_scale_x = torch.cos(torch.linspace(0, walk_length, num_frames) * math.pi).double()
    walk_scale_y = torch.sin(torch.linspace(0, walk_length, num_frames) * math.pi).double()
    noise_x = torch.tensordot(walk_scale_x, walk_noise_x, dims=0)
    noise_y = torch.tensordot(walk_scale_y, walk_noise_y, dims=0)
    batched_noise = noise_x + noise_y

    plot_data = []

    # calculate audio features
    audio_features = []
    for i in range(0, num_frames):
        slice_start = int(i * frame_length * SAMPLING_RATE)
        slice_end = int((i + 1) * frame_length * SAMPLING_RATE)
        audio_slice = audio[slice_start:slice_end]
        value = audio_feature(audio_slice, SAMPLING_RATE)
        audio_features.append(value)

    # pre-processing neural audio features
    if audio_feature_name == 'encodec_feature':
        # could do PCA here to reduce dimensionality to 50
        N, H, W = len(audio_features), 4, 4
        num_components = H*W
        audio_feature_name += "_TSNE" + str(num_components)
        audio_features = TSNE(n_components=num_components, method='exact', perplexity=15).fit_transform(np.array(audio_features))
        # audio_feature_name += "_PCA" + str(num_components)
        # audio_features = PCA(n_components=num_components).fit_transform(np.array(audio_features))
        if num_components == H*W:
            audio_features = audio_features.reshape(-1, H, W)

        # apply smoothing to matrices
        smoothed_audio_features = np.zeros_like(audio_features)
        if audio_features.ndim == 3:  # we're doing a 4x4 thing
            batch_indices = np.arange(N)
            for i in range(H):
                for j in range(W):
                    a = audio_features[:, i:i+1, j:j+1]
                    smoothed, smoothing_name = smoothing_fn(a.squeeze())
                    plot_data.append(smoothed)
                    smoothed_audio_features[:, i:i+1, j:j+1] = smoothed.reshape(N, 1, 1)
                    # selected_elements = audio_features[batch_indices, i, j]
                    # transformed_elements, _ = smoothing_fn(selected_elements)
                    # audio_features[batch_indices, i, j] = transformed_elements
            non_smoothed_audio_features = audio_features
            audio_features = smoothed_audio_features
            audio_features = torch.from_numpy(audio_features).to(wrapper.stream.device, wrapper.stream.dtype)
        elif audio_features.ndim == 2:
            for i in range(num_components):
                a = audio_features[:, i]
                smoothed, smoothing_name = smoothing_fn(a.squeeze())
                plot_data.append(smoothed)
                smoothed_audio_features[:, i] = smoothed
            non_smoothed_audio_features = audio_features
            audio_features = smoothed_audio_features
            audio_features = torch.from_numpy(audio_features).to(wrapper.stream.device, wrapper.stream.dtype)

        if DEBUG:
            # plotting code for 4x4 matrix
            x = [a / FPS for a in range(len(audio_features))]
            # plot just the first feature, comparing smoothed and unsmoothed
            # plt.plot(x, non_smoothed_audio_features[:, :1, :1].squeeze(), label="unsmoothed feature 1")
            # plt.plot(x, plot_data[0], label="smoothed feature 1")
            # plt.title(f"First dim of EnCodec showing smoothed vs. unsmoothed; Sound: {audio_path.stem}\nSmoothing: {smoothing_name}")
            # plt.legend()
            # plt.xlabel("t (seconds)")
            # plt.show()

            for y in plot_data:
                plt.plot(x, y)
            # if audio_features != smoothed_features:
            #     scaled_features = audio_features
            #     plt.plot(x, scaled_features, label='Scaled features')
            plt.title(f"{num_components} dims of EnCodec; Sound: {audio_path.stem}\nSmoothing: {smoothing_name}")
            # plt.legend()
            plt.xlabel("t (seconds)")
            plt.show()
    else:
        non_smoothed_features = audio_features
        # apply smoothing to audio features
        smoothing_name = "no-smoothing"
        if smoothing_fn is not None:
            audio_features = np.array(audio_features)  # this back and forth between python list and np array could be cleaned up
            audio_features, smoothing_name = smoothing_fn(audio_features)
            audio_features = audio_features.tolist()
        smoothed_features = audio_features

        # calculate min/max for scaling
        if bend_function is not None:
            audio_features_max = max(audio_features)
            audio_features_min = min(audio_features)
            bending_function_min, bending_function_max = bending_functions_range[bend_function]
            # rescale audio features to the range that the bending fn expects
            audio_features = [util.scale_range(x, audio_features_min, audio_features_max, bending_function_min, bending_function_max) for x in audio_features]

        # plot audio features
        if DEBUG:
            # plotting code for scalars
            x = [a / FPS for a in range(len(audio_features))]
            plt.plot(x, non_smoothed_features, label='Raw features')
            plt.plot(x, smoothed_features, label='Smoothed features')
            plt.title(f"Audio feature: {audio_feature_name}; Sound: {audio_path.stem}\nSmoothing: {smoothing_name}")
            plt.legend()
            plt.xlabel("t (seconds)")
            plt.show()

    # do prompt interpolations
    prompt_embeddings = []
    for i in range(num_prompts - 1):
        start_prompt, end_prompt = prompts[i], prompts[i + 1]
        start_frame, end_frame = prompt_times[i], prompt_times[i + 1]

        start_embedding = encode_prompt(wrapper, start_prompt)
        end_embedding = encode_prompt(wrapper, end_prompt)

        embeddings = np.linspace(start_embedding.cpu(), end_embedding.cpu(), num=(end_frame - start_frame))
        prompt_embeddings.append(embeddings)
    prompt_embeddings = np.vstack(prompt_embeddings)
    prompt_embeddings = torch.from_numpy(prompt_embeddings).to(wrapper.stream.device)

    # generate frames
    prompt_embedding = prompt_embeddings[0]
    for i in range(num_frames):
        print(f"Frame {i} / {num_frames}")
        # bend is a function that defines how to apply network bending given a latent tensor and audio
        audio_feature_value = audio_features[i]
        print(">>> Audio Feature Value:", audio_feature_value)
        if bend_function is not None:
            bend = bend_function(audio_feature_value)
        else:
            bend = None
        if TXT2IMG:
            try:
                prompt_embedding = prompt_embeddings[i]
            except IndexError:
                # the length of prompt_embeddings is shorter than num_frames, so just use the previous prompt_embedding
                pass
            encoding2img(wrapper, prompt_embedding, batched_noise[i], bend)
        else:
            if i == 0: continue
            guidance_scale = 0.5
            delta = 0.5
            wrapper.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=guidance_scale,
                delta=delta,
            )
            input_image = Image.open(os.path.join(output, f"{i-1:05}.png"))
            image_tensor = wrapper.preprocess_image(input_image)
            output_image = wrapper(image=image_tensor)
            output_image.save(os.path.join(output, f"{i:05}.png"))

    if width != 512 or height != 512:
        dim_string = f"_{height}x{width}"
    else:
        dim_string = ""
    video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{smoothing_name}_{FPS}fps_seed{seed}{dim_string}-StreamDiffusion.mp4"
    # video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{FPS}fps_seed{seed}{dim_string}-StreamDiffusion.mp4"

    counter = 1
    while video_name.exists():
        video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{smoothing_name}_{FPS}fps_seed{seed}{dim_string}-StreamDiffusion{counter}.mp4"
        # video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{FPS}fps_seed{seed}{dim_string}-StreamDiffusion{counter}.mp4"
        counter += 1

    # turn images into video
    ffmpeg_command = ["ffmpeg",
                      "-y",  # automatically overwrite if output exists
                      "-framerate", str(FPS),  # set framerate
                      "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
                      "-i", str(audio_path),  # set audio path
                      "-vcodec", "libx264",
                      "-pix_fmt", "yuv420p",
                      str(video_name)]
    run(ffmpeg_command)

    print(">>> Saved video as", video_name)
    print(">>> Generated {} images".format(num_frames))
    print(">>> Took", util.time_string(time.time() - tic))
    print(">>> Avg time per frame:", (time.time() - tic) / num_frames)
    print("Done.")
