import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import utils.util as util
from utils.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
 
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
print ('=================')
total_steps = (start_epoch-1) * dataset_size + epoch_iter
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
##############
with torch.autograd.set_detect_anomaly(False):
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        # print ('++++')

        for i, data in enumerate(dataset, start=epoch_iter):
            # print ('++---++') 
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            # losses_list, generated = model(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']), \
            #     real_image=  Variable(data['target_rgb']),dif_img=  Variable(data['dif_img']), \
            #      mis_img=  Variable(data['mis_img']), infer=save_fake)  
            # # reference_img , reference_lmark, target_lmark , real_image, warping_ref_img, warping_ref_lmark , ani_img
            losses, generated = model(reference_img =Variable(data['ref_image']),reference_lmark= Variable(data['ref_label']), target_lmark  =  Variable(data['tgt_label']) ,  \
            real_image=  Variable(data['tgt_image']), warping_ref_img =  Variable(data['warping_ref']),  warping_ref_lmark =  Variable(data['warping_ref_lmark']) ,  ani_img =  Variable(data['ani_image']) ,  ani_lmark =  Variable(data['ani_lmark']), infer=save_fake)
            # sum per device losses
            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()

            loss_G.backward()          
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            
            loss_D.backward()        
            optimizer_D.step()      

               ############## Display results and errors ##########
            ### print out errors
            # print   (loss_dict['D_fake'], loss_dict['D_real'],  loss_dict['G_GAN'],  loss_dict.get('G_GAN_Feat',0),  loss_dict.get('G_VGG',0)) 
            errors = {}
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
                #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

                ### display output images
                        
                tmp = []
                for i in range(opt.num_frames):
                    tmp.extend([( 'reference%d'%i, util.tensor2im(data['ref_image'][0,i]))])
                    
                tmp.extend([('target_lmark', util.tensor2im(data['tgt_label'][0, 0])),
                                    ('synthesized_image', util.tensor2im(generated[0].data[0])),
                                    ('raw', util.tensor2im(generated[1].data[0])),
                                    ('att', util.tensor2im(generated[2].data[0])),
                                    ('real_image', util.tensor2im(data['tgt_image'][0, 0])),
                                    ('ani_image', util.tensor2im(data['ani_image'][0, 0])),
                                     ('ani_lmark', util.tensor2im(data['ani_lmark'][0, 0]))])
                    
               
                visuals =  OrderedDict(tmp)  
                visualizer.display_current_results(visuals, epoch, total_steps)
                
            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
        
        # end of epoch 
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()


# from options.train_options import TrainOptions
# from dataset import LRSLmark2rgbDataset
# opt = TrainOptions().parse()
# dataset = LRSLmark2rgbDataset(opt)
# sample = dataset[0]
# print (sample)