# train on grid dataset using base model
#CUDA_VISIBLE_DEVICES=1,0,2  python base_train.py --gpu_ids=0,1,2 --netG=base3  --name base3_grid_1 --dataname=grid --dataroot /home/cxu-serve/p1/common/grid  --batchSize 36 --niter 10000  --nThreads 8 --n_layers_D=3  --num_D=2 --lr=0.00005 --num_frames=1
# CUDA_VISIBLE_DEVICES=3   python base_train.py --model=base4_pretrain --gpu_ids=0 --netG=base4  --name base4_pretrain_grid_1 --dataname=grid --dataroot /home/cxu-serve/p1/common/grid  --batchSize 16 --niter 10000  --nThreads 1 --n_layers_D=3  --num_D=2 --lr=0.00005 --num_frames=1

# CUDA_VISIBLE_DEVICES=0   python base_test.py --model=base1  --gpu_ids=0 --netG=base4  --name base1_grid_32 --dataname=grid --dataroot /home/cxu-serve/p1/common/grid  --batchSize 16 --niter 10000  --nThreads 1 --num_frames=1 # --n_layers_D=3  --num_D=2 --lr=0.00005 --num_frames=32



# CUDA_VISIBLE_DEVICES=0   python base_train.py --gpu_ids=0 --netG=base3  --name base3_grid_32 --dataname=grid --dataroot /home/cxu-serve/p1/common/grid  --batchSize 8 --niter 10000  --nThreads 8 --n_layers_D=3  --num_D=2 --lr=0.00005 --num_frames=32
CUDA_VISIBLE_DEVICES=0   python base_test.py --netG=base1 --gpu_ids=0  --name base1_lrs_1 --dataname=lrs  --batchSize 36   --nThreads 1 --num_frames=1 --which_epoch 160
# train on foceforensic dataset using base model

# CUDA_VISIBLE_DEVICES=7,6,4   python base_train.py --gpu_ids=0,1,2  --netG=base2 --name base2_lrs --dataname=lrs --batchSize 512 --niter 10000  --dataroot  /mnt/Data02/lchen63/lrs/   --batchSize 8 --niter 10000   --n_layers_D=3  --num_D=2 --lr=0.00005 --num_frames=1
# CUDA_VISIBLE_DEVICES=2,1   python base_train.py --gpu_ids=0,1  --netG=base1 --name base1_face_1 --dataname=facefor --dataroot /home/cxu-serve/p1/common/faceforensics/original_sequences/youtube --batchSize 32 --niter 10000    --niter 10000   --n_layers_D=3  --num_D=2 --lr=0.00005 --num_frames=1
# train on lrs dataset using base model
# CUDA_VISIBLE_DEVICES=1,2 python base_train.py --name base1_lrs_1 --gpu_ids=0,1 --model=base1  --dataname=lrs --batchSize 42 --niter 10000  --nThreads 16 --num_frames=1  --n_layers_D=3  --num_D=2 --lr=0.0001 






