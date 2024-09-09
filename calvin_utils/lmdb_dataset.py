import io
import gc
from time import time
import lmdb
from pickle import loads, dumps
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, decode_png

ORIGINAL_STATIC_RES = 200
DEPTH_MAX = 10
DEPTH_MIN = 0

def float_to_png(depth_img):
    if depth_img.max() > DEPTH_MAX or depth_img.min() < DEPTH_MIN:
        raise ValueError("calvin2lmdb: depth_img wrong range")

    # store a depth map in a png file, with shape (3, h, w)
    depth_img = (2**24) * (depth_img - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    depth_img_int = depth_img.to(torch.int32)

    # Extract the top 24 bits
    r = (depth_img_int >> 16) & 0xFF
    g = (depth_img_int >> 8) & 0xFF
    b = depth_img_int & 0xFF

    # Stack channels to get a shape of [3, h, w]
    depth_map_rgb = torch.stack([r, g, b]).to(torch.uint8)

    return encode_png(depth_map_rgb)

def png_to_float(png):
    depth_map_rgb = decode_png(png)
    r, g, b = depth_map_rgb[0].to(torch.int32), depth_map_rgb[1].to(torch.int32), depth_map_rgb[2].to(torch.int32)
    depth_img_int = (r << 16) + (g << 8) + b
    depth_img = depth_img_int / (2**24) * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
    return depth_img

class DataPrefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        with torch.cuda.stream(self.stream):
            for key in self.batch:
                self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time

class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, sequence_length, chunk_size, action_mode, action_dim, has_quant, start_ratio, end_ratio):
        super(LMDBDataset).__init__()
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.has_quant = has_quant
        self.dummy_rgb_static = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES, ORIGINAL_STATIC_RES, dtype=torch.uint8)
        self.dummy_depth_static = torch.zeros(sequence_length, ORIGINAL_STATIC_RES, ORIGINAL_STATIC_RES)
        self.dummy_arm_state = torch.zeros(sequence_length, 6)
        self.dummy_gripper_state = torch.zeros(sequence_length, 2)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, action_dim)
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.lmdb_dir = lmdb_dir
        env = lmdb.open(self.lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio)
            self.end_step = int(dataset_len * end_ratio) - sequence_length - chunk_size
            if self.has_quant:
                quant_ids = loads(txn.get('quant_ids_0'.encode()))
                self.dummy_quant_ids = torch.zeros((sequence_length,)+quant_ids.shape, dtype=quant_ids.dtype)
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_dir, map_size=int(3e12), readonly=False, lock=False)
        self.txn = self.env.begin(write=True)

    def save_quant_ids(self, quant_ids, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()
        idx = idx + self.start_step
        self.txn.put(f'quant_ids_{idx}'.encode(), dumps(quant_ids))
    
    def flush(self):
        self.txn.commit()
        self.txn = self.env.begin(write=True)

    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()

        idx = idx + self.start_step

        if self.has_quant:
            quant_ids = self.dummy_quant_ids.clone()
        rgb_static = self.dummy_rgb_static.clone()
        depth_static = self.dummy_depth_static.clone()
        arm_state = self.dummy_arm_state.clone()
        gripper_state = self.dummy_gripper_state.clone()
        actions = self.dummy_actions.clone()
        mask = self.dummy_mask.clone()

        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        inst_emb = loads(self.txn.get(f'inst_emb_{cur_episode}'.encode())).float()
        for i in range(self.sequence_length):
            if loads(self.txn.get(f'cur_episode_{idx+i}'.encode())) == cur_episode:
                if self.has_quant:
                    quant_ids[i] = loads(self.txn.get(f'quant_ids_{idx+i}'.encode()))
                rgb_static[i] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx+i}'.encode())))
                depth_static[i] = png_to_float(loads(self.txn.get(f'depth_static_{idx+i}'.encode())))
                robot_obs = loads(self.txn.get(f'robot_obs_{idx+i}'.encode()))
                arm_state[i, :6] = robot_obs[:6]
                gripper_state[i, ((robot_obs[-1] + 1) / 2).long()] = 1
                for j in range(self.chunk_size):
                    if loads(self.txn.get(f'cur_episode_{idx+i+j}'.encode())) == cur_episode:
                        mask[i, j] = 1
                        if self.action_mode == 'ee_rel_pose':
                            actions[i, j] = loads(self.txn.get(f'rel_action_{idx+i+j}'.encode()))
                        elif self.action_mode == 'ee_abs_pose':
                            actions[i, j] = loads(self.txn.get(f'abs_action_{idx+i+j}'.encode()))
                        actions[i, j, -1] = (actions[i, j, -1] + 1) / 2
        out = {
            'rgb_static': rgb_static,
            'depth_static': depth_static,
            'inst_emb': inst_emb,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'actions': actions,
            'mask': mask,
        }
        if self.has_quant:
            out['quant_ids'] = quant_ids
        return out

    def __len__(self):
        return self.end_step - self.start_step
