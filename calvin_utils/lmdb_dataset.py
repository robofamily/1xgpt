import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, decode_png

ORIGINAL_STATIC_RES = 200
ORIGINAL_GRIPPER_RES = 84
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
    def __init__(self, lmdb_dir, sequence_length, chunk_size, action_mode, action_dim, load_gripper_image, start_ratio, end_ratio):
        super(LMDBDataset).__init__()
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.dummy_rgb_static = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES, ORIGINAL_STATIC_RES, dtype=torch.uint8)
        self.dummy_depth_static = torch.zeros(sequence_length, ORIGINAL_STATIC_RES, ORIGINAL_STATIC_RES)
        self.load_gripper_image = load_gripper_image
        if load_gripper_image:
            self.dummy_rgb_gripper = torch.zeros(sequence_length, 3, ORIGINAL_GRIPPER_RES, ORIGINAL_GRIPPER_RES, dtype=torch.uint8)
            self.dummy_depth_gripper = torch.zeros(sequence_length, ORIGINAL_GRIPPER_RES, ORIGINAL_GRIPPER_RES)
        self.dummy_arm_state = torch.zeros(sequence_length, 6)
        self.dummy_gripper_state =  torch.zeros(sequence_length, 2)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, action_dim)
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.lmdb_dir = lmdb_dir
        env = lmdb.open(lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio)
            self.end_step = int(dataset_len * end_ratio) - sequence_length - chunk_size
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()

        idx = idx + self.start_step

        rgb_static = self.dummy_rgb_static.clone()
        depth_static = self.dummy_depth_static.clone()
        if self.load_gripper_image:
            rgb_gripper = self.dummy_rgb_gripper.clone()
            depth_gripper = self.dummy_depth_gripper.clone()
        arm_state = self.dummy_arm_state.clone()
        gripper_state = self.dummy_gripper_state.clone()
        actions = self.dummy_actions.clone()
        mask = self.dummy_mask.clone()

        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        inst_token = loads(self.txn.get(f'inst_token_{cur_episode}'.encode()))
        for i in range(self.sequence_length):
            if loads(self.txn.get(f'cur_episode_{idx+i}'.encode())) == cur_episode:
                rgb_static[i] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx+i}'.encode())))
                depth_static[i] = png_to_float(loads(self.txn.get(f'depth_static_{idx+i}'.encode())))
                if self.load_gripper_image:
                    rgb_gripper[i] = decode_jpeg(loads(self.txn.get(f'rgb_gripper_{idx+i}'.encode())))
                    depth_gripper[i] = png_to_float(loads(self.txn.get(f'depth_gripper_{idx+i}'.encode())))
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
            'inst_token': inst_token,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'actions': actions,
            'mask': mask,
        }
        if self.load_gripper_image:
            out['rgb_gripper'] = rgb_gripper
            out['depth_gripper'] = depth_gripper
        return out

    def __len__(self):
        return self.end_step - self.start_step
