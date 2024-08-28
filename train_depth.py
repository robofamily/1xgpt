import os
from pathlib import Path
import math
import json
import copy
from time import time
import argparse
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from calvin_utils.lmdb_dataset import LMDBDataset as LMDBdst_jpeg, DataPrefetcher, DEPTH_MAX, DEPTH_MIN
from Magvit2Dpt import Magvit2Dpt

def visualize(acc, examples):
    for key in examples:
        rgb, real_depth, pred_depth = examples[key]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(rgb.permute(1,2,0).cpu().numpy())
        axs[0].set_title('RGB')
        axs[0].axis('off')

        vmin = min(real_depth.min(),pred_depth.min())
        vmax = max(real_depth.max(), pred_depth.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)

        real_depth = resize(real_depth.unsqueeze(0), pred_depth.shape[-2:])[0]
        axs[1].imshow(real_depth.cpu().numpy(), cmap='viridis', norm=norm)
        axs[1].set_title('Real Depth')
        axs[1].axis('off')

        axs[2].imshow(pred_depth.cpu().numpy(), cmap='viridis', norm=norm)
        axs[2].set_title('Pred Depth')
        axs[2].axis('off')

        cbar = fig.colorbar(sm, ax=axs[1:], orientation='vertical')
        cbar.set_label('Depth Value')

        acc.log({f"{key} dataset": fig})
        plt.close(fig)

def train(acc, train_prefetcher, test_prefetcher, model, optimizer, scheduler, device, args):
    '''
    prof = profile(
        schedule = torch.profiler.schedule(
            wait=20,
            warmup=3,
            active=4,
            repeat=1,
        ),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(save_path/'prof'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    prof.start()
    '''

    train_dataset_len = len(train_prefetcher.loader.dataset)
    test_dataset_len = len(test_prefetcher.loader.dataset)
    eval_steps = train_dataset_len // test_dataset_len
    for epoch in tqdm(range(args.num_epochs), desc=f"train epochs", disable=not acc.is_main_process):
        if epoch % args.save_epochs == 0:
            if epoch != 0:
                acc.wait_for_everyone()
                unwrapped_model = acc.unwrap_model(model)
                if hasattr(unwrapped_model, '_orig_mod'):
                    acc.save(unwrapped_model._orig_mod.state_dict(), save_path/'magvit2dpt_{}.pth'.format(epoch+args.load_epoch))
                else:
                    acc.save(unwrapped_model.state_dict(), save_path/'magvit2dpt_{}.pth'.format(epoch+args.load_epoch))

        batch_metric = {
            'loss': 0,
            'eval_loss': 0,
            'grad_norm_before_clip': 0,
            'dataload_time': 0,
        } 
        avg_metric = copy.deepcopy(batch_metric) # average over batches
        examples = {}
        clock = time()
        batch_idx = 0
        progress_bar = tqdm(range(train_dataset_len//args.bs_per_gpu//acc.num_processes), desc=f"train steps in epoch", disable=not acc.is_main_process)
        batch, batch_metric['dataload_time'] = train_prefetcher.next()
        while batch is not None:
            with acc.accumulate(model):
                model.train()
                optimizer.zero_grad()
                rgb_static = resize(batch['rgb_static'][:, 0], (args.square_resolution, args.square_resolution))
                rgb_static = rgb_static / 127.5 - 1
                pred_depth = model(rgbs=rgb_static)
                loss = F.mse_loss(pred_depth, resize(batch['depth_static'][:, 0], pred_depth.shape[-2:]))
                acc.backward(loss)
                if acc.sync_gradients:
                    batch_metric['grad_norm_before_clip'] = acc.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                batch_metric['loss'] = loss.detach()
                for key in batch_metric:
                    if 'eval' not in key:
                        avg_metric[key] += batch_metric[key] / args.print_steps
                examples['train'] = (batch['rgb_static'][0, 0], batch['depth_static'][0, 0], pred_depth[0].detach())

            if batch_idx % eval_steps == 0:
                with torch.no_grad():
                    model.eval()
                    batch, _ = test_prefetcher.next_without_none()
                    rgb_static = resize(batch['rgb_static'][:, 0], (args.square_resolution, args.square_resolution))
                    rgb_static = rgb_static / 127.5 - 1
                    pred_depth = model(rgbs=rgb_static)
                    batch_metric['eval_loss'] = F.mse_loss(pred_depth, resize(batch['depth_static'][:, 0], pred_depth.shape[-2:]))
                    for key in batch_metric:
                        if 'eval' in key:
                            avg_metric[key] += batch_metric[key] / args.print_steps * eval_steps
                    examples['eval'] = (batch['rgb_static'][0, 0], batch['depth_static'][0, 0], pred_depth[0].detach())

            if batch_idx % args.print_steps == 0 and batch_idx != 0:
                avg_metric['dataload_percent_first_gpu'] = avg_metric['dataload_time'] * args.print_steps / (time()-clock)
                avg_metric['lr'] = scheduler.get_last_lr()[0]
                avg_metric['fps_first_gpu'] = (args.bs_per_gpu * args.print_steps) / (time()-clock)
                clock = time()

                for key in batch_metric:
                    if key != 'dataload_time':
                        avg_metric[key] = acc.gather_for_metrics(avg_metric[key]).mean()

                text = '\nTrain Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, 
                    batch_idx * args.bs_per_gpu * acc.num_processes, 
                    train_dataset_len, 
                    100. * batch_idx * args.bs_per_gpu * acc.num_processes / train_dataset_len, 
                )
                for key in avg_metric:
                    text = text + ' {}: {:.5f}'.format(key, avg_metric[key])
                acc.print(text)
                acc.log(avg_metric)
                if acc.is_main_process:
                    visualize(acc, examples)

                for key in avg_metric:
                    avg_metric[key] = 0 
                scheduler.step()

            batch_idx += 1
            progress_bar.update(1)
            batch, batch_metric['dataload_time'] = train_prefetcher.next()
            '''
            prof.step()
            if batch_idx == 28:
                prof.stop()
            '''

def parse_args():
    parser = argparse.ArgumentParser(description="Train a network that translates MagVit2 latent to DepthAnything Embedding")

    parser.add_argument("--calvin_dir", type=str)
    parser.add_argument("--dpt_ckpt_path", type=str)
    parser.add_argument("--magvit_ckpt_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--load_epoch", type=int)
    parser.add_argument("--square_resolution", default=200, type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--num_warmup_epochs", type=float)
    parser.add_argument("--save_epochs", type=int)
    parser.add_argument("--bs_per_gpu", default=24, type=int)
    parser.add_argument("--lr_max", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=10, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--workers_per_gpu", default=12, type=int)
    parser.add_argument("--print_steps", default=200, type=int)
    parser.add_argument("--freeze_dpt", default=True, type=bool)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Preparation
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    device = acc.device
    train_dataset = LMDBdst_jpeg(
        lmdb_dir=args.calvin_dir, 
        sequence_length=1,
        chunk_size=1,
        action_mode='rel',
        action_dim=7,
        load_gripper_image=False,
        start_ratio = 0,
        end_ratio = 0.9, 
    )
    test_dataset = LMDBdst_jpeg(
        lmdb_dir=args.calvin_dir, 
        sequence_length=1,
        chunk_size=1,
        action_mode='rel',
        action_dim=7,
        load_gripper_image=False,
        start_ratio = 0.9,
        end_ratio = 1, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.bs_per_gpu, # to be flattened in prefetcher  
        num_workers=args.workers_per_gpu,
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.bs_per_gpu, # to be flattened in prefetcher  
        num_workers=args.workers_per_gpu,
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
    ) 
    model = Magvit2Dpt(freeze_dpt=args.freeze_dpt).to(device)
    model.load_magvit(magvit_ckpt_path=args.magvit_ckpt_path)
    model.load_dpt(dpt_ckpt_path=args.dpt_ckpt_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(save_path/'magvit2dpt_{}.pth'.format(args.load_epoch)):
        state_dict = torch.load(save_path/'magvit2dpt_{}.pth'.format(args.load_epoch))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        run_id = json.load(open(save_path/"wandb_id.json", "r"))
        acc.init_trackers(
            project_name="magvit2dpt",
            init_kwargs={"wandb": {"id": run_id, "resume": "allow"}}
        )
        acc.print('load ', save_path/'magvit2dpt_{}.pth'.format(args.load_epoch), '\nmissing ', missing_keys, '\nunexpected ', unexpected_keys)
    else:
        acc.init_trackers(project_name="magvit2dpt")
        run_id = acc.get_tracker("wandb").run.id
        json.dump(run_id, open(save_path/"wandb_id.json", "w"))
        acc.print(save_path/'magvit2dpt_{}.pth'.format(args.load_epoch), 'does not exist. Initialize new checkpoint')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_max, fused=False)
    total_prints_per_epoch = len(train_dataset) // (args.print_steps * args.bs_per_gpu * acc.num_processes)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_prints_per_epoch*args.num_warmup_epochs),
        num_training_steps=args.num_epochs*total_prints_per_epoch,
    )
    model, optimizer, train_loader, test_loader = acc.prepare(
        model, 
        optimizer, 
        train_loader, 
        test_loader, 
        device_placement=[True, True, False, False],
    )
    train_prefetcher = DataPrefetcher(train_loader, device)
    test_prefetcher = DataPrefetcher(test_loader, device)

    # Train
    train(acc, train_prefetcher, test_prefetcher, model, optimizer, scheduler, device, args)
