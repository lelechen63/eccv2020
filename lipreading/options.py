device_ids='0,1'
random_seed = 0
data_type = 'unseen'
data_root= '/mnt/ssd0/dat/lchen63/grid'
vid_padding = 75
txt_padding = 200
batch_size = 2
base_lr = 2e-5
num_workers = 40
max_epoch = 1000
display = 10
test_step = 1000
save_prefix = f'weights/tcn_{data_type}'
is_optimize = True

# weights = 'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
