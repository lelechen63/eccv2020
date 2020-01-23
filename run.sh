# train on foceforensic dataset using base model

CUDA_VISIBLE_DEVICES=3 python base_train.py --name base_facefor --dataname=facefor --dataroot /home/cxu-serve/p1/common/faceforensics/original_sequences/youtube

# train on lrs dataset using base model


CUDA_VISIBLE_DEVICES=3 python base_train.py --name base_lrs


