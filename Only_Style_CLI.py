from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from  src import Controlled_sa_handler
from src.Alpha_search import search_optimal_alpha
from utils.utils import convert_tokens_to_indices
import torch
import numpy as  np
import os
import shutil
import math
from PIL import Image
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Only Style CLI')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--S_ref', type=str, default="A cat", help='Reference subject')
    parser.add_argument('--ref_token', type=str, default="cat", help='Reference subject token')
    parser.add_argument('--S_tgt', type=str, default="A train", help='Target subject')
    parser.add_argument('--tgt_token', type=str, default="train", help='Target subject token')
    parser.add_argument('--style', type=str, default="in 3D rendering style", help='Style')
    parser.add_argument('--precision', type=float, default=0.03, help='Precision')
    parser.add_argument('--output_dir', type=str, default="./output", help='Output directory')
    parser.add_argument('--leak_detection_iter', type=int, default=49, help='Iter within diffusion process to detect leakage')
    args = parser.parse_args()
    
    # init model
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        scheduler=scheduler
    ).to("cuda")
    
    #### Create output directory ####
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    #### change into it
    os.chdir(args.output_dir)
    
     ##### Create metadata directory #####   
    if os.path.exists('./metadata'):
        shutil.rmtree('./metadata')
    os.makedirs('./metadata')
    
    root_dir = os.getcwd()
    
    ref_indices = convert_tokens_to_indices(pipeline, [args.S_ref], [args.ref_token])[0]
    tgt_indices = convert_tokens_to_indices(pipeline, [args.S_tgt], [args.tgt_token])[0]
    
    set_of_prompts = []

    set_of_prompts.append(args.S_ref + " " + args.style)
    set_of_prompts.append(args.S_tgt + " " + args.style)
    
    # This simulates the base StyleAligned process
    alpha = 1.0
                
    # Want to generate the same ref img
    generator = torch.Generator("cuda")
    generator.manual_seed(args.seed)
        
    handler = Controlled_sa_handler.Handler(pipeline)
    sa_args = Controlled_sa_handler.StyleAlignedArgs(share_group_norm=False,
                        share_layer_norm=False,
                        share_attention=True,
                        adain_queries=True,
                        adain_keys=True,
                        adain_values=False,
                        )
    
    #### We infer the ref and the target using the base StyleAligned process for visualization 
    #### Also used to detect the ref subject (this can be performed by just infering the ref image
    #### but here we generate the StyleAligned image pair to illustrate leakage) 
    handler.register(sa_args, root_dir, alpha, ref_indices, tgt_indices, leak_detection_iter = args.leak_detection_iter)
    images = pipeline(set_of_prompts, generator=generator, num_inference_steps=50, ).images
     
    # Save the reference and target images
    images[0].save(f"./Reference.png")
    images[1].save(f"./StyleAligned_Target.png")
    
    # Load a tensor leak from ../../../metadata
    leak = torch.load(os.path.join(root_dir, "metadata/leak.pt"))
    
    # boolean variable indicating content leakage
    flag = (torch.sum(leak) != 0)
    
    ##### Visualize the Content Leakage heatmap #####
    diff = torch.load(os.path.join(root_dir, "metadata/diff.pt"))
    res = int(math.sqrt(diff.shape[0]))
    diff = diff.reshape(res,res).cpu().numpy()

    # Resize the attention map to match the image size (1024x1024)
    attention_map_resized = np.array(Image.fromarray(diff).resize(images[1].size, Image.BILINEAR))

    # Display the image and overlay the "attention map"
    plt.figure(figsize=(10, 10))
    plt.imshow(images[1], cmap='gray')
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.5)  # alpha controls the transparency
    plt.axis('off')
    plt.show()

    # Save the image
    plt.savefig(f"./StyleAligned_Leakage.png", bbox_inches=0)

    # close the figure
    plt.close()
    
    with open(f"./log.txt", 'a') as f:
        f.write(f"Number of leaky patches:  {torch.sum(leak)} " + f' / a = {alpha}\n')
    
    images = search_optimal_alpha(set_of_prompts, handler, pipeline, args.precision, ref_indices, tgt_indices, args.seed, sa_args, root_dir,
                                  args.leak_detection_iter)    
    
    # Save the Only-Style target image
    images[1].save(f"./Only_Style_Target.png")
