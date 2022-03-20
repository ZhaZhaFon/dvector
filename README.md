# D-vector

This is a PyTorch implementation of speaker embedding trained with GE2E loss.
The original paper about GE2E loss could be found here: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)

### 文件组织

```
    dvector/
        data/
            wav2mel.py 音频张量转Mel谱的相关接口
            ge2e_dataset.py Dataset接口
            infinite_dataloader.py DataLoader接口
        modules/
            
        Makefile
        preprocess.py 数据预处理 将数据集音频转为Mel谱
```

### Preprocess training data 训练数据准备

运行脚本前需要先将训练集数据组织为以下形式:
- 同一个说话人的语料在同一目录下(即**说话人目录**)
- 每个**说话人目录**可以有自己的若干子目录(e.g.同一说话人不同场景采集的数据)
- 所有**说话人目录**放在同一个目录下(即**数据集根目录**)  

以VoxCeleb1为例:
```
    your/path/to/voxceleb/
        id10001/
            1zcIwhmdeo4/
                00001.wav
                00002.wav
                ...
            7gWzIy6yIIk/
            ...
        id10002/
        ...
```

运行以下命令将训练集音频转为Mel谱:

```
    python preprocess.py "/your/path/to/voxceleb1/" -o "your/path/to/save/voxceleb1_mel"
```

To use the script provided here, you have to organize your raw data in this way:

- all utterances from a speaker should be put under a directory (**speaker directory**)
- all speaker directories should be put under a directory (**root directory**)
- **speaker directory** can have subdirectories and utterances can be placed under subdirectories

And you can extract utterances from multiple **root directories**, e.g.

```bash
python preprocess.py VoxCeleb1/dev LibriSpeech/train-clean-360 -o preprocessed
```

If you need to modify some audio preprocessing hyperparameters, directly modify `data/wav2mel.py`.
After preprocessing, 3 preprocessing modules will be saved in the output directory:
1. `wav2mel.pt`
2. `sox_effects.pt`
3. `log_melspectrogram.pt`

> The first module `wav2mel.pt` is composed of the second and the third modules.
> These modules were compiled with TorchScript and can be used anywhere to preprocess audio data.  

---
---

## Usage

```python
import torch
import torchaudio

wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector.pt").eval()

wav_tensor, sample_rate = torchaudio.load("example.wav")
mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
```

You can also embed multiple utterances of a speaker at once:

```python
emb_tensor = dvector.embed_utterances([mel_tensor_1, mel_tensor_2])  # shape: (emb_dim)
```

There are 2 modules in this example:
- `wav2mel.pt` is the preprocessing module which is composed of 2 modules:
    - `sox_effects.pt` is used to normalize volume, remove silence, resample audio to 16 KHz, 16 bits, and remix all channels to single channel
    - `log_melspectrogram.pt` is used to transform waveforms to log mel spectrograms
- `dvector.pt` is the speaker encoder

Since all the modules are compiled with [TorchScript](https://pytorch.org/docs/stable/jit.html), you can simply load them and use anywhere **without any dependencies**.

### Pretrianed models & preprocessing modules

You can download them from the page of [*Releases*](https://github.com/yistLin/dvector/releases).

## Evaluate model performance

You can evaluate the performance of the model with equal error rate.
For example, download the official test splits (`veri_test.txt` and `veri_test2.txt`) from [The VoxCeleb1 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and run the following command: 
```bash
python equal_error_rate.py VoxCeleb1/test VoxCeleb1/test/veri_test.txt -w wav2mel.pt -c dvector.pt
```

So far, the released checkpoint was only trained on VoxCeleb1 without any data augmentation.
Its performance on the official test splits of VoxCeleb1 are as following:
| Test Split | Equal Error Rate | Threshold |
| :-:        |:-:               |:-:        |
| veri_test.txt  | 12.0% | 0.222 |
| veri_test2.txt | 11.9% | 0.223 |

## Train from scratch

### Train a model

You have to specify where to store checkpoints and logs, e.g.

```bash
python train.py preprocessed <model_dir>
```

During training, logs will be put under `<model_dir>/logs` and checkpoints will be placed under `<model_dir>/checkpoints`.
For more details, check the usage with `python train.py -h`.

### Use different speaker encoders

By default I'm using 3-layerd LSTM with attentive pooling as the speaker encoder, but you can use speaker encoders of different architecture.
For more information, please take a look at `modules/dvector.py`.

## Visualize speaker embeddings

You can visualize speaker embeddings using a trained d-vector.
Note that you have to structure speakers' directories in the same way as for preprocessing.
e.g.

```bash
python visualize.py LibriSpeech/dev-clean -w wav2mel.pt -c dvector.pt -o tsne.jpg
```

The following plot is the dimension reduction result (using t-SNE) of some utterances from LibriSpeech.

![TSNE result](images/tsne.png)
