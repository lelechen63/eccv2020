# train on foceforensic dataset using base model

CUDA_VISIBLE_DEVICES=3 python base_train.py --name base_facefor --dataname=facefor --debug --dataroot /mnt/Data/lchen63/faceforensics/original_sequences

# train on lrs dataset using base model


CUDA_VISIBLE_DEVICES=3 python base_train.py --name base_lrs


