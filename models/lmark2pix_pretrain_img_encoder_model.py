import numpy as np
import torch
import os
from torch.autograd import Variable
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from collections import OrderedDict

class Lmark2PixHDModel(BaseModel):
    def name(self):
        return 'Lmark2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        input_nc = opt.num_frames * 2 * 3 # or opt.num_frames  * 3

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc  
                       
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 3 + opt.output_nc
           
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

       
        # load networks
        new_state_dict = OrderedDict()
        state_dict = torch.load('/home/cxu-serve/p1/common/weights/zkou2/face_bone.pt')
        for k, v in state_dict.items():
            name =  k[12:] # remove `module.`
            new_state_dict[name] = v
        
        gg = self.netG.img_encoder.state_dict()
        print (gg.keys())
        print ('=================================')
        print (new_state_dict.keys())
        self.netG.img_encoder.load_state_dict(new_state_dict)
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
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
           
            params = list(self.netG.lmark_encoder.parameters()) + list(self.netG.fusion.parameters()) + list(self.netG.decoder.parameters())
                 
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, references = None, target_lmark = None, real_image = None,  infer=False):             
        if target_lmark is not None:
            target_lmark = Variable(target_lmark.data.cuda(non_blocking=True), volatile=infer)      

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda(non_blocking=True))
            
        if references is not None:
            references = Variable(references.data.cuda(non_blocking=True))
        

        return references, target_lmark, real_image 

    def discriminate(self, target_lmark, test_image, use_pool=False):
        input_concat = torch.cat((target_lmark, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, references , target_lmark, real_image,  infer=False):
        # Encode Inputs
        references, target_lmark, real_image = self.encode_input(references, target_lmark, real_image)  

        # Fake Generation        
        fake_image = self.netG.forward(references , target_lmark )

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(target_lmark, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(target_lmark, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((target_lmark, fake_image), dim=1))        
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
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ), None if not infer else fake_image ]

    def inference(self, references, target_lmark,  real_image):
        # Encode Inputs        
        real_image = Variable(real_image) if real_image is not None else None
        references, target_lmark, real_image = self.encode_input(references, target_lmark, real_image, infer=True) 
        # Fake Generation
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(references , target_lmark )
        else:
            fake_image = self.netG.forward(references , target_lmark)
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

class InferenceModel(Lmark2PixHDModel):
    def forward(self, inp):
        references, target_lmark, image = inp
        return self.inference(references, target_lmark, image)
