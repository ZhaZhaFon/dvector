
prepare:
	python3 preprocess.py "/home/zzf/dataset/vox1/wav" -o "/home/zzf/dataset/vox1_mel"

train:
	python3 train.py --gpu "2" --data_dir "/home/zzf/dataset/vox1_mel" --save_dir "/home/zzf/experiment-dvector/dvec_pool-attn" --test_dir "/home/zzf/dataset/vox1/wav_test" --test_txt "/home/zzf/dataset/vox1/veri_test.txt" 
eval:
	python3 equal_error_rate.py --test_dir "/home/zzf/dataset/vox1/wav_test" --test_txt "/home/zzf/dataset/vox1/veri_test.txt" -w "/home/zzf/dataset/vox1_mel/wav2mel.pt" -c "/home/zzf/experiment-dvector/dvec_pool-attn/checkpoints/dvector-epoch299.pt"
visualize:
	python3 visualize.py --gpu "2" --data_dir "/home/zzf/corpus/librispeech/LibriSpeech/test-clean" -w "/home/zzf/dataset/vox1_mel/wav2mel.pt" -c "/home/zzf/experiment-dvector/dvec_pool-attn/checkpoints/dvector-epoch299.pt" -o tsne.jpg