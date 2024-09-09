#!/usr/bin/env python3
import os
import math
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms.functional import resize
from einops import rearrange
from transformers import AutoModel, AutoTokenizer
import lmdb
from pickle import loads
from torchvision.io import decode_jpeg

from magvit2.config import VQConfig
from magvit2.models.lfqgan import VQModel

ORIGINAL_STATIC_RES = 200

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer lmdb dataset format to 1x memmap dataset format")
    parser.add_argument(
        "--in_dir",
        type=str,
        help="Path of directory of the lmdb dataset",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path of directory of the 1x dataset",
    )
    parser.add_argument(
        "--magvit_ckpt_path", type=str, help="Path to the magvit ckpt file"
    )
    parser.add_argument(
        "--square_resolution", default=ORIGINAL_STATIC_RES, type=int, help="Resize the image to a square image with specified resolution"
    )
    parser.add_argument(
        "--max_length", default=4000000, type=int, help="Maximum number of frames in the dataset"
    )
    parser.add_argument(
        "--start_ratio", default=0, type=float,
    )
    parser.add_argument(
        "--end_ratio", default=0, type=float,
    )
    args = parser.parse_args()
    return args

def encode_video_wrapper(video_data, square_res, magvit, batch_size=16):
    """
    video_data: (t, h, w, c)
    """

    encoded_indices = []

    video_data = video_data.to(magvit.device).to(magvit.dtype) / 127.5 - 1
    video_data = resize(video_data, [square_res, square_res])
    for shard_ind in range(math.ceil(len(video_data) / batch_size)):
        batch = video_data[shard_ind * batch_size: (shard_ind + 1) * batch_size]
        with torch.no_grad():
            quant, _, _, _ = magvit.encode(batch)
            indices = magvit.quantize.bits_to_indices(rearrange(((quant + 1) / 2).bool(), "b c h w -> b h w c")).to(torch.int64)
            encoded_indices.append(indices)

    return torch.cat(encoded_indices, dim=0)

def main():
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = "cuda"
    dtype = torch.bfloat16
    magvit = VQModel(VQConfig(), ckpt_path=args.magvit_ckpt_path)
    magvit = magvit.to(device=device, dtype=dtype)

    '''
    model_path = 'Alibaba-NLP/gte-base-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    emb_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    '''

    video_fp = None
    env = lmdb.open(args.in_dir, readonly=True, create=False, lock=False)
    with env.begin() as txn:
        dataset_len = loads(txn.get('cur_step'.encode())) + 1
        start_frame_id = int(dataset_len*args.start_ratio)
        start_ep_idx = loads(txn.get(f'cur_episode_{start_frame_id}'.encode()))
        end_frame_id = int(dataset_len*args.end_ratio)
        last_ep_idx = 0
        for frame_idx in tqdm(range(start_frame_id, end_frame_id, 1), desc="convert frame"):
            rgb_static = decode_jpeg(loads(txn.get(f'rgb_static_{frame_idx}'.encode()))).unsqueeze(0)
            encoded_indices = encode_video_wrapper(rgb_static, args.square_resolution, magvit)[0].cpu().numpy()
            ep_idx = loads(txn.get(f'cur_episode_{frame_idx}'.encode()))
            lang_emb = loads(txn.get(f'inst_emb_{ep_idx}'.encode()))

            if video_fp is None:
                video_fp = np.memmap(os.path.join(args.out_dir, 'video.bin'), dtype=np.uint32, mode='w+', shape=(args.max_length, encoded_indices.shape[1], encoded_indices.shape[1]))
                segment_fp = np.memmap(os.path.join(args.out_dir, 'segment_ids.bin'), dtype=np.int32, mode='w+', shape=(args.max_length,))
                lang_fp = np.memmap(os.path.join(args.out_dir, 'language_emb.bin'), dtype=np.float32, mode='w+', shape=(args.max_length, lang_emb.shape[0]))

            video_fp[frame_idx - start_frame_id] = encoded_indices
            segment_fp[frame_idx - start_frame_id] = ep_idx - start_ep_idx
            lang_fp[ep_idx - start_ep_idx] = lang_emb

            if ep_idx > last_ep_idx:
                last_ep_idx = ep_idx
                video_fp.flush()
                lang_fp.flush()
                segment_fp.flush()
        video_fp.flush()
        lang_fp.flush()
        segment_fp.flush()
    env.close()
    
    meta_json_path = open(os.path.join(args.out_dir, "metadata.json"), "w")
    metadata = {
        "token_dtype": "uint32", 
        "s": encoded_indices.shape[1], 
        "h": encoded_indices.shape[1], 
        "w": encoded_indices.shape[1], 
        "vocab_size": VQConfig.codebook_size, 
        "hz": 30, 
        "num_images": end_frame_id - start_frame_id,
        "tokenizer_ckpt": str(args.magvit_ckpt_path),
        "lang_emb_dim": lang_emb.shape[0],
    }
    json.dump(metadata, meta_json_path)

if __name__ == "__main__":
    main()