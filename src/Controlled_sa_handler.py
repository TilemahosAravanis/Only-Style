from __future__ import annotations
from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
import einops
import math
import gc
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

T = torch.Tensor

@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True,
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1. # Scales down the shared attention map
    shared_score_shift: float = 0. # != 0 for shifted self attention
    only_self_level: float = 0. # Percentage of attention layers that do not implement shared self attention

def expand_first(feat: T, scale=1.,) -> T:
    b = feat.shape[0] # B
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1) # (2, 1, h, T, C)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:]) # (2, b/2, h, T, C)
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1) # (2, b/2, h, T, C)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1) # (2, b/2, h, T, C)
    return feat_style.reshape(*feat.shape) # (B, h, T, C)

def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale) # (B, h, T, C)
    return torch.cat((feat, feat_style), dim=dim) # (B, h, 2*T, C)

def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

# Applies instance normalization with the first batch image
def adain(feat: T) -> T:
    # calc mean and std across Patch dimension
    feat_mean, feat_std = calc_mean_std(feat) # (B, h, 1, C)
    feat_style_mean = expand_first(feat_mean) # (B, h, 1, C)
    feat_style_std = expand_first(feat_std) # (B, h, 1, C)
    feat = (feat - feat_mean) / feat_std # (B, h, T, C)
    feat = feat * feat_style_std + feat_style_mean # (B, h, T, C)
    return feat

# Annotate as True the percentage*100% greater values
def percentile_threshold_cross_attn_map(tensor: T, percentage: float):   
    # Flatten the tensor
    flattened_tensor = tensor.view(-1).to(torch.float)

    # Calculate the threshold value based on the given percentage
    threshold_value = torch.quantile(flattened_tensor, 1 - percentage)

    # Apply thresholding
    thresholded_tensor = tensor > threshold_value

    return thresholded_tensor.flatten()

# To retrieve the cross attention probabilities
def dummy_cross_attn_weights(image_features, text_embeddings, to_q, to_k):  
    query = to_q(image_features) # (B, T1, C) 
    key = to_k(text_embeddings) # (B, T2, C)
    
    batch_size = key.shape[0]
    
    if image_features.shape[1] == 1024:
        num_heads = 20
    else:
        num_heads = 10

    inner_dim = image_features.shape[-1] # C
    head_dim = inner_dim // num_heads # C//H == h (H == num_heads)

    query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2).to(torch.float64) # (B, H, T1, h)
    key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2).to(torch.float64) # (B, H, T2, h)

    scale_factor = 1 / math.sqrt(query.size(-1)) 
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    key[3] = key[2]
    attn_weight_2 = query @ key.transpose(-2, -1) * scale_factor

    return attn_weight, attn_weight_2

# To retrevieve the cosine similarity scores used for leakage detection
def dummy_self_attn_weights(image_features, key_embedds, to_q, to_k):
    query = image_features.to(torch.float64) # (B, T1, C) 
    key = key_embedds.to(torch.float64) # (B, T2, C)

    # Normalize the keys
    key_norm = torch.norm(key, dim=-1, keepdim=True) # (B, T2, 1)
    key = key / key_norm # (B, T2, C)

    # Normalize the queries
    query_norm = torch.norm(query, dim=-1, keepdim=True) # (B, T1, 1)
    query = query / query_norm # (B, T1, C)

    attn_weight = query @ key.transpose(-2, -1)

    return attn_weight

def attention_based_style_alignment_scaling(key, ref_patches, scale):
    if (key.shape[2]//2 == 4096):
        # Upsample ref_patches to 64x64
        ref = ref_patches.reshape(32,32).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).flatten()
    else:
        ref = ref_patches
        
    key[1, :, key.shape[2]//2:][:, ref] *= scale.item()
    key[3, :, key.shape[2]//2:][:, ref] *= scale.item()
    
    return key

# Extracts the patches used to perform the subject pooling
def extract_subject_patches(ref_tokens, target_tokens, cross_attn_map):
    ref_subject_softmaxed = cross_attn_map[0,:,ref_tokens].sum(axis = -1)
    target_subject_softmaxed = cross_attn_map[1,:,target_tokens].sum(axis = -1)

    # kmeans clustering with optimal number of clusters
    ref_kmeans = optimal_kmeans(ref_subject_softmaxed.unsqueeze(1).cpu().numpy())
    ref_subject_mask_1 = torch.zeros(1024, dtype=torch.bool).to("cuda")
    ref_subject_mask_1[ref_kmeans.labels_ == ref_kmeans.cluster_centers_.argmax()] =  1.0

    target_kmeans = optimal_kmeans(target_subject_softmaxed.unsqueeze(1).cpu().numpy())
    target_subject_mask_1 = torch.zeros(1024, dtype=torch.bool).to("cuda")
    target_subject_mask_1[target_kmeans.labels_ == target_kmeans.cluster_centers_.argmax()] =  1.0

    # percentile threshold the two tensors
    ref_subject_mask_2 = percentile_threshold_cross_attn_map(ref_subject_softmaxed, 0.12)
    target_subject_mask_2 = percentile_threshold_cross_attn_map(target_subject_softmaxed, 0.12)

    # logical and of the two masks
    ref_subject_mask = ref_subject_mask_1 & ref_subject_mask_2
    target_subject_mask = target_subject_mask_1 & target_subject_mask_2

    # If the mask sums to zero
    if torch.sum(ref_subject_mask) < 0.02 * 1024:
        ref_subject_mask = percentile_threshold_cross_attn_map(ref_subject_softmaxed, 0.12)
    
    if torch.sum(target_subject_mask) < 0.02 * 1024:
        target_subject_mask = percentile_threshold_cross_attn_map(target_subject_softmaxed, 0.12)
        
    # remove isolated pixels
    ref_subject_mask = remove_isolated_pixels(ref_subject_mask)
    target_subject_mask = remove_isolated_pixels(target_subject_mask)
    
    ref_subject = torch.where(ref_subject_mask, ref_subject_softmaxed, 0.)
    target_subject = torch.where(target_subject_mask, target_subject_softmaxed, 0.)

    # Normalize the scores
    ref_subject = ref_subject/torch.sum(ref_subject)
    target_subject = target_subject/torch.sum(target_subject)

    return ref_subject, target_subject

# Detects content leakage (saves relevant tensors in the metadata directory)
def Content_leakage_detection(self_attn_map):
    
    # Retrieve target image attn map for the target subject text <token>s
    heat_self = self_attn_map[1, :, 0]

    # Retrieve target image attn map for the ref subject text <token>s
    heat_self_2 = self_attn_map[3, :, 0]

    # Combine the two masks (numpy tensors) with or
    mask = torch.logical_or(heat_self > 0.4, heat_self_2 > 0.4)

    # soft leakage values
    leak = torch.where(mask, heat_self_2, -1.) - torch.where(mask, heat_self, 0.)
    
    leak = leak > 0.1
    
    # remove isolated pixels
    leak = remove_isolated_pixels(leak)

    diff = torch.where(mask, heat_self_2, -1.) - torch.where(mask, heat_self, 0.)
    diff = torch.where(leak, diff, 0.)

    return diff, leak

def optimal_kmeans(data, min_k=3, max_k=5):
    best_score = -1
    best_kmeans = None
    scores = []

    # Try different values of k from 3 to max_k
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit_predict(data)

        # Calculate the silhouette score for the current number of clusters
        score = silhouette_score(data, kmeans.labels_)
        scores.append(score)

        # Update the best score and best k if the current one is better
        if score > best_score:
            best_score = score
            best_kmeans = kmeans

    return best_kmeans

# remove pixels with no neighbors 
def remove_isolated_pixels(tensor: T):
    # convert tensor to np.array and reshape it to 9x9  (32x32)
    array = tensor.cpu().numpy().reshape(32,32)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(array.astype(np.uint8) , -1, kernel)
    mask = neighbor_count > 1
    
    array = array * mask
    
    return torch.tensor(array).to("cuda").flatten()
  
class ClearCache:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.empty_cache()
        
class AttentionAggregator:
    def __init__(self):
        self.running_attn_mean = None  # This will hold the running mean of the attention maps
        self.count = 0  # Counter to keep track of the number of the attention maps added

    def aggregate_attn_map(self, attn_map: T):
        if self.running_attn_mean is None:
            # Initialize the running mean with the first attention map
            self.running_attn_mean = attn_map.clone()
        else:
            # Update the running mean
            self.running_attn_mean = self.running_attn_mean * (self.count / (self.count + 1)) + attn_map / (self.count + 1)
        self.count += 1

    def get_running_attn_mean(self):
        return self.running_attn_mean

class DefaultAttentionProcessor(nn.Module):
    """
    Processor for implementing default attention.
    """
    def default_call(
        self,
        attn: attention_processor.Attention,
        hidden_states,  # (B, T1, C1)
        encoder_hidden_states = None, # (B, T2, C2) 
        attention_mask = None,
        temb = None,
        **kwargs
    ):
        residual = hidden_states # (B, T1, C1)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim # number of dimensions of the input tensor

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2) # (B, T1, C1)
            # flattens the last 2 dimensions and transpose sequence length with embedding size

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        ) # Shape of the encoder_hidden_states

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) # (B, T1, C)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if ("attn2" in self.name) and (self.iter.value() == (self.leak_detection_iter - 1)) and (hidden_states.shape[1] == 1024):
            cross_attn_weight, cross_attn_weight_2 = dummy_cross_attn_weights(hidden_states, encoder_hidden_states, attn.to_q, attn.to_k)
            
            avg_head_softmaxed = torch.softmax(cross_attn_weight, dim = -1).sum(axis = 1)/cross_attn_weight.shape[1]
            avg_head_softmaxed_2 = torch.softmax(cross_attn_weight_2, dim = -1).sum(axis = 1)/cross_attn_weight_2.shape[1]

            # Save cross attention maps
            self.Iter_cross_attn_aggregator.aggregate_attn_map(torch.cat((avg_head_softmaxed[2:], avg_head_softmaxed_2[2:])))

        if ("attn2" in self.name) and (self.scale == 1.0) and (hidden_states.shape[1] == 1024):
            cross_attn_weight, _ = dummy_cross_attn_weights(hidden_states, encoder_hidden_states, attn.to_q, attn.to_k)
            
            avg_head_softmaxed = torch.softmax(cross_attn_weight, dim = -1).sum(axis = 1)/cross_attn_weight.shape[1]
            avg_head_softmaxed = avg_head_softmaxed[2, :, self.ref_tokens].sum(axis = -1)

            # Save cross attention maps
            self.Global_cross_attn_aggregator.aggregate_attn_map(avg_head_softmaxed)
            
            # Save ref_patches in metadata directory
            if (self.iter.value() == 50):
                torch.save(self.Global_cross_attn_aggregator.get_running_attn_mean() , os.path.join(self.root_dir, "metadata/ref_patches.pt"))


        key = attn.to_k(encoder_hidden_states) # (B, T2, C) 
        value = attn.to_v(encoder_hidden_states) # (B, T2, C) 

        inner_dim = key.shape[-1] # C
        head_dim = inner_dim // attn.heads # C//H == h (H == num_heads)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, T1, h)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, T2, h)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, T2, h)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = nnf.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # (B, T1, C)
        hidden_states = hidden_states.to(query.dtype) 
        # linear proj
        hidden_states = attn.to_out[0](hidden_states) # (B, T1, C1)
        # dropout
        hidden_states = attn.to_out[1](hidden_states) # (B, T1, C1)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual # (B, T1, C1)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states # (B, T1, C1)
    

    def __init__(self, name: str, iter: int, scale: float, ref_tokens, target_tokens, leak_detection_iter: int,
                 Global_cross_attn_aggregator: AttentionAggregator, Iter_cross_attn_aggregator: AttentionAggregator, root_dir: str):
        super().__init__()

        self.name = name
        self.iter = iter
        self.scale = scale
        self.ref_tokens = ref_tokens
        self.target_tokens = target_tokens
        self.leak_detection_iter = leak_detection_iter
        self.root_dir = root_dir
        self.Global_cross_attn_aggregator = Global_cross_attn_aggregator
        self.Iter_cross_attn_aggregator = Iter_cross_attn_aggregator

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.default_call(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)


class SharedAttentionProcessor(DefaultAttentionProcessor):
    """
    Processor for implementing shared attention.
    """

    def shifted_scaled_dot_product_attention(self, attn: attention_processor.Attention, query: T, key: T, value: T) -> T:
        logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
        logits[:, :, :, query.shape[2]:] += self.shared_score_shift
        probs = logits.softmax(-1)
        return torch.einsum('bhqk,bhkd->bhqd', probs, value)

    def shared_call(
            self,
            attn: attention_processor.Attention,
            hidden_states, # (B, T1, C1)
            encoder_hidden_states=None, # (B, T1, C1)
            attention_mask=None,
            **kwargs
    ):
    
        residual = hidden_states # (B, T1, C1)
        input_ndim = hidden_states.ndim # number of dimensions of the input tensor
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2) # (B, T1, C1)
            # flattens the last 2 dimensions and transpose sequence length with embedding size
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if (self.name == 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'):
            # increment the iteration counter
            self.iter.increment()

            if (self.iter.value() == self.max_iter):
                # Detect content Leakage
                diff, leak = Content_leakage_detection(self.Iter_self_attn_aggregator.get_running_attn_mean(),)
                
                # Save the leak tensor in the metadata directory as a .pt file
                torch.save(leak, os.path.join(self.root_dir, "metadata/leak.pt"))
                # Save the diff tensor in the metadata directory as a .pt file
                torch.save(diff, os.path.join(self.root_dir, "metadata/diff.pt"))
                
        
        if (self.iter.value() == self.leak_detection_iter) and (self.name == 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'):

            ref_subject, target_subject = extract_subject_patches(self.ref_tokens, self.target_tokens, self.Iter_cross_attn_aggregator.get_running_attn_mean())
            
            # Save the subject representation pooling values in the respective aggregators
            self.ref_subject_aggregator.aggregate_attn_map(ref_subject)
            self.target_subject_aggregator.aggregate_attn_map(target_subject)
            
        if (self.iter.value() == self.leak_detection_iter) and (hidden_states.shape[1] == 1024):
            # Load ref_subject and target_subject from the respective aggregators
            ref_subject = self.ref_subject_aggregator.get_running_attn_mean()
            target_subject = self.target_subject_aggregator.get_running_attn_mean()
            
            image_features_before_projection_attn1 = hidden_states.to(torch.float64)

            # Pooling of the visual embeddings into one subject visual embedding
            ref_visual_embedding = (ref_subject @ image_features_before_projection_attn1[0, :, :]).unsqueeze(0)
            target_visual_embedding = (target_subject @ image_features_before_projection_attn1[1, :, :]).unsqueeze(0)
            
            ref_visual_embedding = torch.cat([ref_visual_embedding.unsqueeze(0)]*hidden_states.shape[0])
            target_visual_embedding = torch.cat([target_visual_embedding.unsqueeze(0)]*hidden_states.shape[0])

            self_attn_weight_2 = dummy_self_attn_weights(image_features_before_projection_attn1, ref_visual_embedding, attn.to_q, attn.to_k)
            self_attn_weight = dummy_self_attn_weights(image_features_before_projection_attn1, target_visual_embedding, attn.to_q, attn.to_k)
            
            # Aggregate the self attention maps in the self attention aggregator
            self.Iter_self_attn_aggregator.aggregate_attn_map(torch.cat((self_attn_weight[:2], self_attn_weight_2[:2])))

        query = attn.to_q(hidden_states) # (B, T1, C)
        key = attn.to_k(hidden_states) # (B, T1, C)
        value = attn.to_v(hidden_states) # (B, T1, C)
        inner_dim = key.shape[-1] # C
        head_dim = inner_dim // attn.heads # C//H == h (H == num_heads)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, T1, h)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, T1, h)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, T1, h)
        # if self.step >= self.start_inject:
        
        if self.adain_queries:
            query = adain(query) # (B, H, T1, h)
        if self.adain_keys:
            key = adain(key) # (B, H, T1, h)
        if self.adain_values:
            value = adain(value) # (B, H, T1, h)  
        if self.share_attention:
        
            # Implement Shared attention
            key = concat_first(key, -2, scale=self.shared_score_scale)
            value = concat_first(value, -2)
            
            ### attention based style transfer scaling ####
            if (self.scale != 1.0):     
                key = attention_based_style_alignment_scaling(key, self.ref_patches, torch.tensor(self.scale))

            if self.shared_score_shift != 0:
                hidden_states = self.shifted_scaled_dot_product_attention(attn, query, key, value,) # !No support for saving attn maps
            else:
                hidden_states = nnf.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        else:
            hidden_states = nnf.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            ) # (B, H, T1, h)

        # hidden_states = adain(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # (B, T1, C)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) # (B, T1, C1)
        # dropout
        hidden_states = attn.to_out[1](hidden_states) # (B, T1, C1)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual # (B, T1, C1)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states # (B, T1, C1)
        
    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        if self.full_attention_share:
            b, n, d = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, '(k b) n d -> k (b n) d', k=2)
            hidden_states = super().__call__(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                             attention_mask=attention_mask, **kwargs)
            hidden_states = einops.rearrange(hidden_states, 'k (b n) d -> (k b) n d', n=n)
        else:
            hidden_states = self.shared_call(attn, hidden_states, hidden_states, attention_mask, **kwargs)

        return hidden_states

    def __init__(self, style_aligned_args: StyleAlignedArgs, name: str, iter, scale: float, ref_tokens, target_tokens, leak_detection_iter: int, max_iter: int,
                 ref_patches: T, Iter_cross_attn_aggregator: AttentionAggregator, Iter_self_attn_aggregator: AttentionAggregator, 
                 ref_subject_aggregator: AttentionAggregator, target_subject_aggregator: AttentionAggregator, root_dir: str = '.'):
        
        super().__init__(name, iter, scale, ref_tokens, target_tokens, None, leak_detection_iter, Iter_cross_attn_aggregator, root_dir)
        self.share_attention = style_aligned_args.share_attention
        self.adain_queries = style_aligned_args.adain_queries
        self.adain_keys = style_aligned_args.adain_keys
        self.adain_values = style_aligned_args.adain_values
        self.full_attention_share = style_aligned_args.full_attention_share
        self.shared_score_scale = style_aligned_args.shared_score_scale
        self.shared_score_shift = style_aligned_args.shared_score_shift

        self.name = name
        self.iter = iter
        self.max_iter = max_iter
        self.scale = scale
        self.ref_tokens = ref_tokens
        self.target_tokens = target_tokens
        self.leak_detection_iter = leak_detection_iter
        self.root_dir = root_dir
        self.ref_patches = ref_patches
        self.Iter_cross_attn_aggregator = Iter_cross_attn_aggregator
        self.Iter_self_attn_aggregator = Iter_self_attn_aggregator
        self.ref_subject_aggregator = ref_subject_aggregator
        self.target_subject_aggregator = target_subject_aggregator

# A class to store the iteration of the model, it should have an init, an increment method and a value method
class Iteration:
    def __init__(self):
        self.iter = 0

    def increment(self):
        self.iter += 1
        
    def value(self):
        return self.iter

def _get_switch_vec(total_num_layers, level):
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool)
    to_flip = level > .5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers)
    vec = torch.arange(total_num_layers)
    vec = vec % (total_num_layers // num_switch)
    vec = vec == 0
    if to_flip:
        vec = ~vec
    return vec


def init_attention_processors(pipeline: StableDiffusionXLPipeline, scale: float, ref_tokens, target_tokens, 
                              leak_detection_iter: int, style_aligned_args: StyleAlignedArgs | None = None, 
                              root_dir: str = '.', num_inference_steps: int = 50):
    '''
    A function that modifies the attention processors of the unet of a StableDiffusionXLPipeline object.
    '''
    # Basically the unet.attn_processors is a dictionary { 'layer_name' : 'class_for_fwd_pass', ...}
    attn_procs = {}
    unet = pipeline.unet
    number_of_only_self, number_of_shared_self, number_of_cross = 0, 0, 0
    num_self_layers = len([name for name in unet.attn_processors.keys() if 'attn1' in name])
    
    # Instances of the AttentionAggregator class to store the running mean of the attention maps
    Global_cross_attn_aggregator = AttentionAggregator()
    Iter_cross_attn_aggregator = AttentionAggregator()
    Iter_self_attn_aggregator = AttentionAggregator()
    ref_subject_aggregator = AttentionAggregator()
    target_subject_aggregator = AttentionAggregator()
    
    ref_patches = None
    
    # if c != 1.0 load the ref_patches tensor, which are the patches to retain in the attention based transfer
    if scale != 1.0:
        # load ref patches 
        ref_patches = torch.load(os.path.join(root_dir, f'metadata/ref_patches.pt'))
        
        # Cluster the ref_patches tensor and keep the cluster with the highest average value
        ref_kmeans = optimal_kmeans(ref_patches.unsqueeze(1).cpu().numpy(), min_k=2, max_k=2)
        ref_patches = torch.zeros(1024, dtype=torch.bool).to("cuda")
        ref_patches[ref_kmeans.labels_ == ref_kmeans.cluster_centers_.argmax()] = 1.0

        # remove isolated pixels from the ref_patches tensor
        ref_patches = remove_isolated_pixels(ref_patches)
        
        # iniialize a 3 by 3 numpy kernel for morphological operations, data type is uint8
        kernel = np.ones((3, 3), dtype=np.uint8)

        # dilate the ref_patches tensor
        ref_patches = cv2.morphologyEx(ref_patches.cpu().numpy().astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # convert the ref_patches tensor to a torch tensor and move it to the GPU
        ref_patches = torch.tensor(ref_patches).to("cuda").squeeze().to(torch.bool)
        
    # Instance of Iteration class to store the iteration of the model
    iter = Iteration()
    
    if style_aligned_args is None:
        only_self_vec = _get_switch_vec(num_self_layers, 1)
    else:
        only_self_vec = _get_switch_vec(num_self_layers, style_aligned_args.only_self_level)
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attention = 'attn1' in name
        if is_self_attention:
        
            number_of_only_self += 1
            if style_aligned_args is None or only_self_vec[i // 2]: 
                # Set the attention processor of the self attention layer as default
                attn_procs[name] = DefaultAttentionProcessor(name, iter, scale, ref_tokens, target_tokens, leak_detection_iter,
                                                             Global_cross_attn_aggregator, Iter_cross_attn_aggregator, root_dir)
            else:
                # Set the attention processor of the self attention layer as shared
                attn_procs[name] = SharedAttentionProcessor(style_aligned_args, name, iter, scale, ref_tokens, target_tokens, leak_detection_iter,
                                                            num_inference_steps , ref_patches, Iter_cross_attn_aggregator, Iter_self_attn_aggregator, 
                                                            ref_subject_aggregator, target_subject_aggregator, root_dir)
                number_of_shared_self += 1
        else:
            # Set the attention processor of the cross attention layer as default
            number_of_cross += 1
            attn_procs[name] = DefaultAttentionProcessor(name, iter, scale, ref_tokens, target_tokens, leak_detection_iter,
                                                         Global_cross_attn_aggregator, Iter_cross_attn_aggregator, root_dir)

    unet.set_attn_processor(attn_procs)


def register_shared_norm(pipeline: StableDiffusionXLPipeline,
                         share_group_norm: bool = True,
                         share_layer_norm: bool = True, ):
    '''
    A function that modifies the normalization layers of a StableDiffusionXLPipeline object.
    '''

    def register_norm_forward(norm_layer: nn.GroupNorm | nn.LayerNorm) -> nn.GroupNorm | nn.LayerNorm:
        #  Modify the forward pass of a single layer
        if not hasattr(norm_layer, 'orig_forward'):
            setattr(norm_layer, 'orig_forward', norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_
        return norm_layer

    def get_norm_layers(pipeline_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]):
        if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
            norm_layers_['layer'].append(pipeline_)
        if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
            norm_layers_['group'].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {'group': [], 'layer': []}
    get_norm_layers(pipeline.unet, norm_layers)
    return [register_norm_forward(layer) for layer in norm_layers['group']] + [register_norm_forward(layer) for layer in
                                                                               norm_layers['layer']]

class Handler:
    '''
    A class that modifies a StableDiffusionXLPipeline object. Specifically the normalization layers and attention processors.
    '''

    def register(self, style_aligned_args: StyleAlignedArgs , root_dir: str ,scale: float, ref : list, target: list, leak_detection_iter: int, 
                 num_inference_steps: int = 50):
        # Change the normalization layers
        self.norm_layers = register_shared_norm(self.pipeline, style_aligned_args.share_group_norm,
                                                style_aligned_args.share_layer_norm)
        # Change the attention layers
        init_attention_processors(self.pipeline, scale, ref, target, leak_detection_iter, style_aligned_args, root_dir, num_inference_steps)

    def remove(self):
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers = []
        init_attention_processors(self.pipeline, None)

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        
        self.pipeline = pipeline # Pass the pipeline as a parameter of the class
        self.norm_layers = []
