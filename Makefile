
prepare:
	python3 preprocess.py "/home/zzf/dataset/vox1/wav" -o "/home/zzf/dataset/vox1_mel"

train:
	rm -rf /home/zzf/experiment-dvector/dvec_pool-attn
	python3 train.py --gpu "2" --data_dir "/home/zzf/dataset/vox1_mel" --save_dir "/home/zzf/experiment-dvector/dvec_pool-attn" --test_dir "/home/zzf/dataset/vox1/wav_test" --test_txt "/home/zzf/dataset/vox1/veri_test.txt" 