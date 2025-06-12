import os
import torch

def search_optimal_alpha(set_of_prompts, handler, pipeline, precision, ref_tokens, target_tokens, seed, sa_args, root_dir, leak_detection_iter):
    alpha_low = 0
    alpha_high = 1

    while alpha_high - alpha_low > precision:
        alpha = (alpha_low + alpha_high) / 2

        # Want to generate the same ref img
        generator = torch.Generator("cuda")
        generator.manual_seed(seed)
        
        handler.register(sa_args, root_dir, alpha, ref_tokens, target_tokens, leak_detection_iter = leak_detection_iter)
        images = pipeline(set_of_prompts, generator=generator, num_inference_steps=50, ).images

        # Load a tensor leak from metadata directory
        leak = torch.load(os.path.join(root_dir, "metadata/leak.pt"))
        # assign a boolean variable leak as False only if all the sum of the tensor is 0
        flag = (torch.sum(leak) != 0)

        with open(f"./log.txt", 'a') as f:
            f.write(f"Number of leaky patches:  {torch.sum(leak)} " + f' / a = {alpha}\n')

        if flag:
            alpha_high = alpha
        else:
            alpha_low = alpha

    if alpha == alpha_high:

        alpha = alpha_low

        # Want to generate the same ref img
        generator = torch.Generator("cuda")
        generator.manual_seed(seed)
         
        handler.register(sa_args, root_dir, alpha, ref_tokens, target_tokens, leak_detection_iter = leak_detection_iter)   
        images = pipeline(set_of_prompts, generator=generator, num_inference_steps=50, ).images

        # Load a tensor leak from ../../../metadata
        leak = torch.load(os.path.join(root_dir, "metadata/leak.pt"))
        # assign a boolean variable leak as False only if all the sum of the tensor is 0
        flag = (torch.sum(leak) != 0)
        
        with open(f"./log.txt", 'a') as f:
            f.write(f"Number of leaky patches:  {torch.sum(leak)} " + f' / a = {alpha}\n')

    return images