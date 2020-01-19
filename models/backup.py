
class GlobalGenerator(nn.Module):
    def __init__(self,input_nc, output_nc, pad_type='reflect', norm_layer=nn.BatchNorm2d, ngf = 64, opt = None):
        super(GlobalGenerator, self).__init__()        
        activ = 'relu'    
        self.deform = opt.use_deform
        self.ft = opt.use_ft
        self.attention =  not opt.no_att
        self.ft_freeze = opt.ft_freeze
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True) ]
        ### downsample
        model += [Conv2dBlock(64, 128, 4, 2, 1,           # 128, 128, 128 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(128, 128, 4, 2, 1,           # 128, 64 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(128, 256, 4, 2, 1,           # 256 32 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(256, 256, 4, 2, 1,           # 256 16
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(256, 512, 4, 2, 1,           # 512 8
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(512, 512, 4, 2, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]


        self.lmark_ani_encoder = nn.Sequential(*model)
        model = []
        ###  adain resnet blocks
        model += [ResBlocks(2, 512, norm  = 'adain', activation=activ, pad_type='reflect')]

        ### upsample         
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]    # 512, 8 , 8 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 512, 16 , 16 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 32, 32 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(256, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 64, 64 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(256, 128, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 128, 128, 128 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(128, 64, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 64, 256, 256 
        if not self.attention:
            model += [Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]

            self.decoder = nn.Sequential(*model)
        else:
            self.decoder = nn.Sequential(*model)

        self.alpha_conv = Conv2dBlock(64, 1, 7, 1, 3,
                                   norm='none',
                                   activation='sigmoid',
                                   pad_type=pad_type)

        

        self.rgb_conv = Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)
    
        self.embedder = Embedder()
        self.mlp = MLP(512,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')
        if not self.deform:
            model = [nn.ReflectionPad2d(3), nn.Conv2d(6, 32, kernel_size=7, padding=0), norm_layer(32), nn.ReLU(True) ]
            ### downsample
            model += [Conv2dBlock(32, 64, 4, 2, 1,           # 128, 128, 128 
                                        norm= 'in',
                                        activation=activ,
                                        pad_type=pad_type)]

            model += [nn.ConvTranspose2d(64, 64,kernel_size=4, stride=(2),padding=(1)),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(True)
            ]
            self.foregroundNet = nn.Sequential(*model)
            
        else:
            self.conv_first =  nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(6, 64, kernel_size=7, padding=0), norm_layer(64), nn.ReLU(False) ])
            self.off2d_1 = nn.Sequential(*[  nn.Conv2d(64, 18 * 8, kernel_size=3, stride =1, padding=1), nn.InstanceNorm2d(18 * 8), nn.ReLU(False)])
            self.def_conv_1 = DeformConv(64, 64, 3,stride =1, padding =1, deformable_groups= 8)
            self.def_conv_1_norm = nn.Sequential(*[  nn.InstanceNorm2d(64), nn.ReLU(False)])

            self.off2d_2 = nn.Sequential(*[  nn.Conv2d(64, 18 * 8, kernel_size=3, stride =1, padding=1), nn.InstanceNorm2d(18 * 8), nn.ReLU(False)])

            self.def_conv_2 = DeformConv(64, 128, 3,stride =1, padding =1, deformable_groups= 8)
            self.def_conv_2_norm =  nn.Sequential(*[  nn.InstanceNorm2d(128), nn.ReLU(False)])

            self.off2d_3 = nn.Sequential(*[  nn.Conv2d(128, 18 * 8, kernel_size=3, stride =1,padding=1), nn.InstanceNorm2d(18 * 8), nn.ReLU(False)])

            self.def_conv_3 = DeformConv( 128, 64, 3,stride =1, padding =1, deformable_groups= 8)
            self.def_conv3_norm = nn.Sequential(*[  nn.InstanceNorm2d(64), nn.ReLU(False)])

        self.beta  = Conv2dBlock(128, 1, 7, 1, 3,
                                    norm='none',
                                    activation='sigmoid',
                                    pad_type=pad_type)

    def forward(self, references, g_in, similar_img, cropped_similar_img):
        dims = references.shape

        references = references.reshape( dims[0] * dims[1], dims[2], dims[3], dims[4]  )
        e_vectors = self.embedder(references).reshape(dims[0] , dims[1], -1)
        if self.ft :
            if self.ft_freeze:
                e_vectors = e_vectors.detach()
        e_hat = e_vectors.mean(dim = 1)
        feature = self.lmark_ani_encoder(g_in)
        # Decode
        adain_params = self.mlp(e_hat)
        assign_adain_params(adain_params, self.decoder)
        if not self.attention:
            return [self.decoder(feature)]
        I_feature = self.decoder(feature)

        I_hat = self.rgb_conv(I_feature)        
        ani_img = g_in[:,3:,:,:]
        ani_img.data = ani_img.data.contiguous()
        alpha = self.alpha_conv(I_feature)
        face_foreground = (1 - alpha) * ani_img + alpha * I_hat
        if not self.deform:
            foreground_feature = self.foregroundNet( torch.cat([ani_img, similar_img], 1) ) 
            forMask_feature = torch.cat([foreground_feature, I_feature ], 1)
            beta = self.beta(forMask_feature)
            # mask = ani_img> -0.9
            # similar_img[mask] = -1 
            image = (1- beta) * cropped_similar_img + beta * face_foreground 
        else:   
            # with bug, can not be solved yet !!!!!!!!!!!!!!!!!!!!
            feature = torch.cat([ani_img, cropped_similar_img], 1)
            fea = self.conv_first(feature)
            offset_1 = self.off2d_1(fea)

            fea = self.def_conv_1(fea, offset_1)
            fea = self.def_conv_1_norm(fea)

            offset_2 = self.off2d_2(fea)

            fea = self.def_conv_2(fea, offset_2)
            fea = self.def_conv_2_norm(fea)

            offset_3 = self.off2d_3(fea)
            
            fea = self.def_conv_3(fea, offset_3)
            foreground_feature = self.def_conv3_norm(fea)

            forMask_feature = torch.cat([foreground_feature, I_feature ], 1)
            beta = self.beta(forMask_feature)

            image = (1- beta) * cropped_similar_img + beta * face_foreground
        

        return [image, cropped_similar_img, face_foreground, beta, alpha, I_hat]


class GlobalGenerator_mfcc(nn.Module):
    def __init__(self,output_nc, pad_type='reflect', norm_layer=nn.BatchNorm2d, ngf = 64):
        super(GlobalGenerator_mfcc, self).__init__()        
        activ = 'relu'    
        model = [nn.ReflectionPad2d(3), nn.Conv2d(6, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True) ]
        ### downsample
        model += [Conv2dBlock(64, 128, 4, 2, 1,           # 128, 128, 128 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(128, 128, 4, 2, 1,           # 128, 64 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(128, 256, 4, 2, 1,           # 256 32 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(256, 256, 4, 2, 1,           # 256 16
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(256, 512, 4, 2, 1,           # 512 8
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(512, 512, 4, 2, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]


        self.lmark_ani_encoder = nn.Sequential(*model)
        model = []
        ###  adain resnet blocks
        model += [ResBlocks(2, 512, norm  = 'adain', activation=activ, pad_type='reflect')]

        ### upsample         
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]    # 512, 8 , 8 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 512, 16 , 16 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 32, 32 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(256, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 64, 64 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(256, 128, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 128, 128, 128 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(128, 64, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 64, 256, 256 
        model += [Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.decoder = nn.Sequential(*model)


        self.embedder = Embedder()


        self.mlp = MLP(512,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')


    def forward(self, references, target_lmark, target_ani, mfcc):
        dims = references.shape
        references = references.reshape( dims[0] * dims[1], dims[2], dims[3], dims[4]  )
        e_vectors = self.embedder(references).reshape(dims[0] , dims[1], -1)
        e_hat = e_vectors.mean(dim = 1)

        g_in = torch.cat([target_lmark, target_ani], 1)

        feature = self.lmark_ani_encoder(g_in)

        # Decode
        adain_params = self.mlp(e_hat)
        assign_adain_params(adain_params, self.decoder)
        image = self.decoder(feature)
        return image
