#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize speaker embeddings."""

from argparse import ArgumentParser
from pathlib import Path
from warnings import filterwarnings

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from librosa.util import find_files
from sklearn.manifold import TSNE
from tqdm import tqdm

import os

def visualize(data_dirs, wav2mel_path, checkpoint_path, output_path, gpu):
    """Visualize high-dimensional embeddings using t-SNE."""

    print(f'# 使用GPU: {gpu}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'# 加载模型 {checkpoint_path}')
    wav2mel = torch.jit.load(wav2mel_path)
    dvector = torch.jit.load(checkpoint_path).eval().to(device)

    print("[INFO] model loaded.")

    n_spkrs = 0
    paths, spkr_names, mels = [], [], []

    for data_dir in data_dirs:
        print(f'# 遍历目录 {data_dir}...')
        data_dir_path = Path(data_dir)
        for spkr_dir in [x for x in data_dir_path.iterdir() if x.is_dir()]:
            n_spkrs += 1
            audio_paths = find_files(spkr_dir)
            spkr_name = spkr_dir.name
            for audio_path in audio_paths:
                paths.append(audio_path)
                spkr_names.append(spkr_name)

    print(f'# 读取数据...')
    for audio_path in tqdm(paths, ncols=0, desc="Preprocess"):
        wav_tensor, sample_rate = torchaudio.load(audio_path)
        with torch.no_grad():
            mel_tensor = wav2mel(wav_tensor, sample_rate)
        mels.append(mel_tensor)

    print(f'# 推理...')
    embs = []
    for mel in tqdm(mels, ncols=0, desc="Embed"):
        with torch.no_grad():
            emb = dvector.embed_utterance(mel.to(device))
            emb = emb.detach().cpu().numpy()
        embs.append(emb)

    print(f'# 计算T-SNE...')
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    transformed = tsne.fit_transform(embs)

    print("[INFO] embeddings transformed.")

    data = {
        "dim-1": transformed[:, 0],
        "dim-2": transformed[:, 1],
        "label": spkr_names,
    }

    print(f'# T-SNE画图并保存到{output_path}...')
    plt.figure()
    sns.scatterplot(
        x="dim-1",
        y="dim-2",
        hue="label",
        palette=sns.color_palette(n_colors=n_spkrs),
        data=data,
        legend="full",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path)
    
    print('\ndone.')


if __name__ == "__main__":
    filterwarnings("ignore")
    PARSER = ArgumentParser()
    PARSER.add_argument("--data_dirs", type=str, nargs="+")
    PARSER.add_argument("-w", "--wav2mel_path", required=True)
    PARSER.add_argument("-c", "--checkpoint_path", required=True)
    PARSER.add_argument("-o", "--output_path", required=True)
    # 新增
    PARSER.add_argument("--gpu", required=True)
    
    print('')
    print('# 解析parser运行参数')
    
    visualize(**vars(PARSER.parse_args()))
