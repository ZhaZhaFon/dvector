#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train d-vector."""

# original codebase: https://github.com/yistLin/dvector
# modified and re-distributed by Zifeng Zhao @ Peking University
# 2022.03

import json
from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from itertools import count
from multiprocessing import cpu_count
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data
import modules
import os
import equal_error_rate
import numpy as np

def train(
    data_dir,
    save_dir,
    n_speakers,
    n_utterances,
    seg_len,
    save_every,
    valid_every,
    decay_every,
    batch_per_valid,
    n_workers,
    comment,
    gpu,
    max_epoch,
    test_dir,
    test_txt,
):
    """Train a d-vector network."""

    # setup job name
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    job_name = f"{start_time}_{comment}" if comment is not None else start_time

    # setup checkpoint and log dirs
    print(f'# 实验保存路径 {save_dir}')
    checkpoints_path = Path(save_dir) / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(save_dir) / "logs")

    # create data loader, iterator
    print(f'# 读取json {Path(data_dir, "metadata.json")}')
    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    print(f'# 加载data.GE2EDataset...')
    dataset = data.GE2EDataset(data_dir, metadata["speakers"], n_utterances, seg_len)
    print(f'# 数据集划分 spk_train/spk_valid={len(dataset) - n_speakers}/{n_speakers}...')
    trainset, validset = random_split(dataset, [len(dataset) - n_speakers, n_speakers])
    # 无限重复采样
    train_loader = data.InfiniteDataLoader(
        trainset,
        batch_size=n_speakers,
        num_workers=n_workers,
        collate_fn=data.collate_batch,
        drop_last=True,
    )
    valid_loader = data.InfiniteDataLoader(
        validset,
        batch_size=n_speakers,
        num_workers=n_workers,
        collate_fn=data.collate_batch,
        drop_last=True,
    )
    train_iter = data.infinite_iterator(train_loader)
    valid_iter = data.infinite_iterator(valid_loader)

    # display training infos
    assert len(trainset) >= n_speakers
    assert len(validset) >= n_speakers
    print(f"[INFO] Use {len(trainset)} speakers for training.")
    print(f"[INFO] Use {len(validset)} speakers for validation.")

    print(f'# 使用GPU: {gpu}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # build network and training tools
    print(f'# 加载模型')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dvector = modules.AttentivePooledLSTMDvector(
        dim_input=metadata["n_mels"],
        seg_len=seg_len,
    ).to(device)
    dvector = torch.jit.script(dvector)
    criterion = modules.GE2ELoss().to(device)
    optimizer = SGD(list(dvector.parameters()) + list(criterion.parameters()), lr=0.01)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=0.95)
    
    print('')
    print('### START TRAINING ###')
    batch_size = n_speakers * n_utterances
    voxceleb1_train_utt = 68224
    steps_per_epoch = voxceleb1_train_utt // batch_size

    # record training infos
    pbar = tqdm(total=steps_per_epoch, ncols=0, desc="Train")
    running_train_loss, running_grad_norm = deque(maxlen=steps_per_epoch), deque(maxlen=steps_per_epoch)
    running_valid_loss = deque(maxlen=batch_per_valid)

    # start training
    for epoch in range(max_epoch):
        
        tqdm.write('')
        tqdm.write(f'# epoch {epoch} - train...')
        pbar.reset()

        for step in count(start=1):
            
            if step > steps_per_epoch:
                break
            
            batch = next(train_iter).to(device)
            embds = dvector(batch).view(n_speakers, n_utterances, -1)
            loss = criterion(embds)

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(dvector.parameters()) + list(criterion.parameters()),
                max_norm=3,
                norm_type=2.0,
            )
            dvector.embedding.weight.grad *= 0.5
            dvector.embedding.bias.grad *= 0.5
            criterion.w.grad *= 0.01
            criterion.b.grad *= 0.01

            optimizer.step()

            running_train_loss.append(loss.item())
            running_grad_norm.append(grad_norm.item())
            avg_train_loss = sum(running_train_loss) / len(running_train_loss)
            avg_grad_norm = sum(running_grad_norm) / len(running_grad_norm)

            pbar.update(1)
            pbar.set_postfix(loss=avg_train_loss, grad_norm=avg_grad_norm)

        #if step % valid_every == 0:
        if True:  
        
            # 交叉验证
            for _ in range(batch_per_valid):
                batch = next(valid_iter).to(device)
                with torch.no_grad():
                    embd = dvector(batch).view(n_speakers, n_utterances, -1)
                    loss = criterion(embd)
                    running_valid_loss.append(loss.item())

            avg_valid_loss = sum(running_valid_loss) / len(running_valid_loss)

            #tqdm.write(f"Valid: epoch={epoch}, error={avg_valid_loss:.1f}")
            writer.add_scalar("train/loss", avg_train_loss, epoch)
            writer.add_scalar("valid/error", avg_valid_loss, epoch)
            writer.add_scalar("learning_rate", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            
            tqdm.write(f'# loss={avg_train_loss}, error={avg_valid_loss}, lr='+str(optimizer.state_dict()['param_groups'][0]['lr']))            

            #if step % save_every == 0:
            if epoch % 20 == 0:
                
                ckpt_path = checkpoints_path / f"dvector-epoch{epoch}.pt"
                dvector.cpu()
                dvector.save(str(ckpt_path))
                dvector.to(device)
                
                eer, thr = equal_error_rate.get_eer(test_dir=test_dir, 
                                                    test_txt=test_txt,
                                                    wav2mel_path=os.path.join(data_dir, "wav2mel.pt"), 
                                                    checkpoint_path=str(ckpt_path),
                                                    )
                tqdm.write(f'# testEER={np.round(eer*100,2)}%, threshold={thr}')
                writer.add_scalar("test/EER", eer, epoch)
        scheduler.step()  
    
    ckpt_path = checkpoints_path / f"dvector-epoch{epoch}.pt"
    dvector.cpu()
    dvector.save(str(ckpt_path))
    dvector.to(device)
    
    print('')
    print('### TRAINING COMPLETED ###')
    from IPython import embed
    embed()
    
    print('\ndone.')


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--data_dir", type=str) # 数据集(Mel谱)路径
    PARSER.add_argument("--save_dir", type=str) # 实验保存路径
    # batch_size = n_speakers * n_utterances = 64 * 4 = 256
    PARSER.add_argument("-n", "--n_speakers", type=int, default=64) # 一个batch中speaker数目
    PARSER.add_argument("-m", "--n_utterances", type=int, default=4) # 轮到的说话人每人每次随机抽取n_utterance条进入batch 语料不足n_utterance条的说话人将被滤除
    PARSER.add_argument("--seg_len", type=int, default=160) # 长度不足seg_len(帧)的音频将被滤除
    PARSER.add_argument("--save_every", type=int, default=10000)
    PARSER.add_argument("--valid_every", type=int, default=1000)
    PARSER.add_argument("--decay_every", type=int, default=10)
    PARSER.add_argument("--batch_per_valid", type=int, default=10)
    PARSER.add_argument("--n_workers", type=int, default=cpu_count()) # 线程
    PARSER.add_argument("--comment", type=str)
    # 新增
    PARSER.add_argument("--gpu", type=str, default="0") # 所用GPU
    PARSER.add_argument("--max_epoch", type=int, default=300) # epoch
    PARSER.add_argument("--test_dir", type=str) # 测试集路径
    PARSER.add_argument("--test_txt", type=str) # 测试列表路径
    
    print('')
    print('# 解析parser运行参数')
    
    train(**vars(PARSER.parse_args()))
