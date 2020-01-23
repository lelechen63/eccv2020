# train on grid dataset using base model

CUDA_VISIBLE_DEVICES=2   python base_train.py --gpu_ids=0  --name base_grid --dataname=grid --dataroot /home/cxu-serve/p1/common/grid --debug  # --batchSize 128 --niter 10000  


# train on foceforensic dataset using base model

#CUDA_VISIBLE_DEVICES=7,6,4   python base_train.py --gpu_ids=0,1,2  --name base_facefor --dataname=facefor --batchSize 128 --niter 10000  --dataroot  /mnt/Data/lchen63/faceforensics/original_sequences

# train on lrs dataset using base model


#CUDA_VISIBLE_DEVICES=3 python base_train.py --name base_lrs


