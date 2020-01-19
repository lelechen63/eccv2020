import numpy as np
import torch
import os
from torch.autograd import Variable
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


#### with landmark + 3d ani
class Lmark2RGBModel1(BaseModel):
    def name(self):
        return 'base1'
    
    def init_loss_filter(self, use_gan_loss, use_vgg_loss, use_face_loss, use_pix_loss):
        flags = (use_gan_loss, use_vgg_loss, use_gan_loss,  use_face_loss, use_pix_loss)
        def loss_filter(g_gan, g_vgg, d_loss_list, g_cnt, g_pix):
            return [l for (l,f) in zip((g_gan,g_vgg,d_loss_list, g_cnt, g_pix),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        input_nc = opt.num_frames * 2 * 3  # example image + example landmark + target landmark        
        ##### define networks        
        # Generator network

        self.netG = networks.define_G(input_nc = input_nc, output_nc =opt.output_nc,netG = opt.netG, \
            pad_type='reflect',norm = opt.norm, ngf = opt.ngf, opt= opt , gpu_ids=self.gpu_ids)             

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc 
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, \
                                          opt.num_D, not opt.ganFeat_loss, opt=opt, gpu_ids=self.gpu_ids)
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
            self.loss_filter = self.init_loss_filter( not opt.no_gan_loss, not opt.no_vgg_loss, not opt.no_face_loss , not opt.no_pixel_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            if not opt.no_pixel_loss:             
                self.criterionPix = networks.PixLoss(self.gpu_ids)

            if not opt.no_face_loss:             
                self.criterionCNT = networks.LossCnt(self.opt)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_VGG','D_losslist', 'G_CNT', 'G_PIX')

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
                  
            # self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            
            if opt.use_ft and opt.ft_freeze:
                for param in self.netG.embedder.parameters():
                    param.requires_grad = False
                self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),  lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))  
            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, references = None, target_lmark = None,  real_image = None, dif_img = None, mis_img = None,  infer=False):             

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda(non_blocking=True))
        
        if references is not None:
            references = Variable(references.data.cuda(non_blocking=True))

        if dif_img is not None:
            dif_img = Variable(dif_img.data.cuda(non_blocking=True))
        
        if mis_img is not None:
            mis_img = Variable(mis_img.data.cuda(non_blocking=True))

        if target_lmark is not None:
            target_lmark = Variable(target_lmark.data.cuda(non_blocking=True))
        if infer == False:
            return references, target_lmark, real_image, dif_img, mis_img
        else:
            return references, target_lmark, real_image , dif_img, mis_img
    def discriminate(self,  reference, lmark , test_image, use_pool=False):
            return self.netD.forward(reference, lmark , test_image.detach())

    def forward(self, references, target_lmark, real_image, dif_img, mis_img, infer=False):
        # Encode Inputs
        references, target_lmark, real_image , dif_img, mis_img = \
        self.encode_input(references = references, target_lmark = target_lmark, real_image = real_image , dif_img = dif_img, mis_img =mis_img, infer= infer)  

        # Fake Generation
        if self.opt.mode == 'base':
            fake_list = self.netG.forward( references, target_lmark )
        # if self.attention:
        fake_image = fake_list[0]
        # Fake Detection and Loss
        if not self.opt.no_gan_loss:
            ##### if it is using lstm, currently, my discriminator does not support lstm operation, we reshape to bactch size
            [_ , _, real_fake_score] = self.discriminate( references , target_lmark, fake_image, use_pool=False)
            loss_D_list = []
            loss_D_fake = self.criterionGAN(real_fake_score, False)   
            loss_D_list.append(loss_D_fake)


            # Real Detection and Loss        
            [matching_score , identity_score, pred_real] = self.discriminate( references , target_lmark, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_list.append(loss_D_real)
            if not self.opt.no_mismatch:
                loss_D_real_matching = self.criterionGAN(matching_score, True)
                loss_D_real_identity = self.criterionGAN(identity_score, True)
                loss_D_list.append(loss_D_real_matching)
                loss_D_list.append(loss_D_real_identity)

            if not self.opt.no_mismatch:
                # mismatch  img & landmark       
                [matching_score , identity_score, _ ] = self.discriminate( references , target_lmark, mis_img)
                loss_D_real_mismatching = self.criterionGAN(matching_score, False)
                loss_D_real_misidentity = self.criterionGAN(identity_score, True)

                loss_D_list.append(loss_D_real_mismatching)
                loss_D_list.append(loss_D_real_misidentity)

            if not self.opt.no_dif:
                # different identity  img        
                [matching_score , identity_score, _ ] = self.discriminate( references , target_lmark, dif_img)
                loss_D_real_difmatching = self.criterionGAN(matching_score, False)
                loss_D_real_difidentity = self.criterionGAN(identity_score, False)

                loss_D_list.append(loss_D_real_difmatching)
                loss_D_list.append(loss_D_real_difidentity)

            [matching_score , identity_score, pred_fake] = self.netD.forward( references , target_lmark, fake_image)         
            loss_G_list = []
            loss_G_GAN = self.criterionGAN(pred_fake, True)    
            loss_G_list.append(loss_G_GAN)
            loss_G_GAN_matching =  self.criterionGAN(matching_score, True)    
            loss_G_list.append(loss_G_GAN_matching)
            loss_G_GAN_identity =  self.criterionGAN(identity_score, True)
            loss_G_list.append(loss_G_GAN_identity)
        else:
            loss_G_list = 0
            loss_D_list = 0
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        loss_G_CNT = 0
        if not self.opt.no_face_loss:
            loss_G_CNT = self.criterionCNT(real_image, fake_image) 

        loss_G_PIX= 0
        if not self.opt.no_pixel_loss:
            # print ('...............', type(real_image),  type(fake_image) )
            loss_G_PIX = self.criterionPix(real_image, fake_image) 
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_list, loss_G_VGG, loss_D_list, loss_G_CNT, loss_G_PIX ), None if not infer else fake_list ]

    def inference(self, references, target_lmark,  real_image ):
        # Encode Inputs        
        real_image = Variable(real_image) if real_image is not None else None

        references, target_lmark,  real_image , _, _  = self.encode_input(references, target_lmark, real_image , None, None, infer=True)

        # Fake Generation           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_list  = self.netG.forward( references, target_lmark)
        else:
            fake_list = self.netG.forward( references, target_lmark)
        return fake_list


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
       

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
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
        references, target_lmark, image = inp
        return self.inference(references, target_lmark, image)
