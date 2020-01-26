# train on grid dataset using base model

CUDA_VISIBLE_DEVICES=2,1   python base_train.py --gpu_ids=0,1  --name base_grid --dataname=grid --dataroot /home/cxu-serve/p1/common/grid  --batchSize 36 --niter 10000  --nThreads 8 --n_layers_D=3  --num_D=2 --lr=0.00005 --num_frames=1

# train on foceforensic dataset using base model

#CUDA_VISIBLE_DEVICES=7,6,4   python base_train.py --gpu_ids=0,1,2  --name base_facefor --dataname=facefor --batchSize 128 --niter 10000  --dataroot  /mnt/Data/lchen63/faceforensics/original_sequences

# train on lrs dataset using base model
#CUDA_VISIBLE_DEVICES=3 python base_train.py --name base_lrs


