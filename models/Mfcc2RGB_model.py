### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Mfcc2RGBModel1(BaseModel):
    def name(self):
        return 'audio_base1'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_face_loss, use_pix_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, use_face_loss, use_pix_loss)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_cnt, g_pix):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake, g_cnt, g_pix),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        input_nc = 6 #(lmark + ani), put is together
        ##### define networks        
        # Generator network
        self.netG = networks.define_G(output_nc =opt.output_nc,netG = opt.netG, pad_type='reflect',norm = opt.norm, gpu_ids=self.gpu_ids)          

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
           
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_face_loss , not opt.no_pixel_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            if not opt.no_pixel_loss:             
                self.criterionPix = networks.PixLoss(self.gpu_ids)

            if not opt.no_face_loss:             
                self.criterionCNT = networks.LossCnt(self.opt)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'G_CNT', 'G_PIX')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
                  
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, references = None, target_lmark = None, target_ani = None, real_image = None, audio = None infer=False):             
        
        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda(non_blocking=True))
        else:
            print ('=======================')
        if references is not None:
            references = Variable(references.data.cuda(non_blocking=True))
        else:
            print ('=======================')
        if target_lmark is not None:
            target_lmark = Variable(target_lmark.data.cuda(non_blocking=True))
        else:
            print ('=======================')
        if target_ani is not None:
            target_ani = Variable(target_ani.data.cuda(non_blocking=True))
        else:
            print ('=======================')

        if audio is not None:
            audio = Variable(audio.data.cuda(non_blocking=True))

        
        return references, target_lmark, target_ani, real_image, audio

    def discriminate(self,  target_lmark, target_ani, test_image, audio, use_pool=False):
        input_concat = torch.cat(( target_lmark, target_ani, test_image.detach()), dim=1)
        
        return self.netD.forward(input_concat, audio)

    def forward(self, references, target_lmark, target_ani, real_image, audio,  infer=False):
        # Encode Inputs
        references, target_lmark, target_ani, real_image, audio  = self.encode_input(references, target_lmark, target_ani, real_image, audio, infer)  

        # Fake Generation
        
        fake_image = self.netG.forward(references, target_lmark, target_ani, audio)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate( target_lmark, target_ani, fake_image, audio, use_pool=False)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate( target_lmark, target_ani, real_image, audio)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat(( target_lmark, target_ani, fake_image), dim=1), audio)        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        loss_G_CNT = 0
        if not self.opt.no_face_loss:
            loss_G_CNT = self.criterionCNT(real_image, fake_image) 

        loss_G_PIX= 0
        if not self.opt.no_pixel_loss:
            loss_G_PIX = self.criterionPix(real_image, fake_image) 
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_CNT, loss_G_PIX ), None if not infer else fake_image ]

    def inference(self, references, target_lmark, target_ani, image, audio):
        # Encode Inputs        
        image = Variable(image) if image is not None else None
        references, target_lmark, target_ani, real_image, audio  = self.encode_input(Vreferences, target_lmark, target_ani, image , audio, infer=True)

        # Fake Generation           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(references, target_lmark, target_ani, audio)
        else:
            fake_image = self.netG.forward(references, target_lmark, target_ani, audio)
        return fake_image


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
       

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel1(Lmark2RGBModel1):
    def forward(self, inp):
        references, target_lmark, target_ani, image, audio = inp
        return self.inference(references, target_lmark, target_ani, image, audio)

        




