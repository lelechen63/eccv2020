import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
import os
import imp
from .vgg import Cropped_VGG19

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)      
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def define_G(input_nc, output_nc, ngf, netG , n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm) 

    if netG == 'global':
        netG = GlobalGenerator1( input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer) 

    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm) 
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class PixLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(PixLoss, self).__init__()        
        self.criterion = nn.L1Loss()

    def forward(self, x, y):              
        loss =   self.criterion(x, y)        
        return loss
##############################################################################
# Generator
##############################################################################



class GlobalGenerator1(nn.Module):
     # the most simple network, the input is I_0 + L_0 + L_i, the output is I_i. We concate all inputs in channel and encode, decode it.
    def __init__(self,input_nc, output_nc, ngf = 64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,  pad_type='reflect'):
        super(GlobalGenerator1, self).__init__()        
        activation = nn.ReLU(True)     
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation ]
        ### downsample
        model += [nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),   # 128, 128, 128 
                      norm_layer(ngf  * 2), activation]
        
        model += [nn.Conv2d(ngf * 2 , ngf  * 2, kernel_size=3, stride=2, padding=1),   # 128, 64 
                      norm_layer(ngf  * 2), activation]

        model += [nn.Conv2d(ngf * 2 , ngf  * 4, kernel_size=3, stride=2, padding=1),   # 256 32 
                      norm_layer(ngf  * 4), activation]

        model += [nn.Conv2d(ngf * 4 , ngf  * 4, kernel_size=3, stride=2, padding=1),   # 256 16 
                      norm_layer(ngf  * 4), activation]

        model += [nn.Conv2d(ngf * 4 , ngf  * 8, kernel_size=3, stride=2, padding=1),   # 512 8
                      norm_layer(ngf  * 8), activation]

        model += [nn.Conv2d(ngf * 8 , ngf  * 8, kernel_size=3, stride=2, padding=1),   # 512 4
                      norm_layer(ngf  * 8), activation]
     

        self.img_encoder = nn.Sequential(*model)

        model = []
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation ]
        ### downsample
        model += [nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),   # 128, 128, 128 
                      norm_layer(ngf  * 2), activation]
        
        model += [nn.Conv2d(ngf * 2 , ngf  * 2, kernel_size=3, stride=2, padding=1),   # 128, 64 
                      norm_layer(ngf  * 2), activation]

        model += [nn.Conv2d(ngf * 2 , ngf  * 4, kernel_size=3, stride=2, padding=1),   # 256 32 
                      norm_layer(ngf  * 4), activation]

        model += [nn.Conv2d(ngf * 4 , ngf  * 4, kernel_size=3, stride=2, padding=1),   # 256 16 
                      norm_layer(ngf  * 4), activation]

        model += [nn.Conv2d(ngf * 4 , ngf  * 8, kernel_size=3, stride=2, padding=1),   # 512 8
                      norm_layer(ngf  * 8), activation]

        model += [nn.Conv2d(ngf * 8 , ngf  * 8, kernel_size=3, stride=2, padding=1),   # 512 4
                      norm_layer(ngf  * 8), activation]
     
        self.lmark_encoder = nn.Sequential(*model)

        model = [nn.Conv2d(ngf * 16 , ngf  * 8, kernel_size=3, stride=1, padding=1),   # 512 4
                      norm_layer(ngf  * 8), activation]
        self.fusion = nn.Sequential(*model)

        model = []
        ###  adain resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf  * 8, padding_type=pad_type, activation=activation, norm_layer=norm_layer)]

        ### upsample  
        model += [nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(ngf * 8), activation] # 512, 8 , 8 

        
        
        model += [nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(ngf * 4), activation] # 256, 16
        
        model += [nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(ngf * 4), activation] # 256, 32

        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(ngf * 2), activation] # 128, 64

        model += [nn.ConvTranspose2d(ngf * 2, ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(ngf ), activation] #  64, 128
        
        model += [nn.ConvTranspose2d(ngf , ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(ngf ), activation] #  64, 256


        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]   

        self.decoder = nn.Sequential(*model)
    
    

    def forward(self, reference, lmark ):
       
        ref_feature = self.img_encoder( reference)

        lmark_feature = self.lmark_encoder( lmark)

        current = torch.cat([ref_feature, lmark_feature], dim = 1)
        feature = self.fusion(current)
        
        return self.decoder(feature)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64,   n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        # for gg in result:
        #     print ('====')
        #     for g in gg:
        #         print (g.shape)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = 1# int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]    #64, 128

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            if n % 2 ==0 :
                nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)      

class MisDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(MisDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = 1  # int(np.ceil((kw-1.0)/2))

        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]    #64, 128

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            if n % 2 ==0 :
                nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)


    def forward(self, input):
        reference = input[:,:-6] 
        lmark = input[:,-6: -3 ]
        img = input[:, -3:] 
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)      

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class AT_net(nn.Module):
    def __init__(self):
        super(AT_net, self).__init__()
        norm_layer = nn.BatchNorm2d
        activation=nn.ReLU(True)
        ngf = 64
        model = []
        model += [nn.Conv2d(1 , ngf  , kernel_size=3, stride=1, padding=1),   # 64, 28, 12 
                      norm_layer(ngf ), activation]

        model += [nn.Conv2d( ngf, ngf * 2  , kernel_size=3, stride=2, padding=1),   # 128, 14, 6 
                      norm_layer(ngf * 2), activation]
        
        model += [nn.Conv2d( ngf * 2, ngf * 2  , kernel_size=3, stride=1, padding=1),   # 128, 14, 6 
                      norm_layer(ngf * 2), activation]
        
        model += [nn.Conv2d( ngf * 2, ngf * 4  , kernel_size=3, stride=1, padding=1),   # 256 7, 3
                      norm_layer(ngf * 4), activation]

        model += [nn.Conv2d( ngf * 4, ngf * 4  , kernel_size=3, stride=2, padding=1),   # 256 7, 3
                      norm_layer(ngf * 4), activation]
        self.audio_eocder = nn.Sequential( * model  )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(256 *  7 * 3,2048),
            nn.ReLU(True),
            nn.Linear(2048,512),
            nn.ReLU(True),
       
            )
        self.lstm = nn.LSTM(512,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256, 68 * 2)
            )

    def forward(self, example_landmark, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            # features = torch.cat([example_landmark_f,  current_feature], 1)
            lstm_input.append(current_feature)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        fc_out   = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in) + example_landmark)
        return torch.stack(fc_out, dim = 1)