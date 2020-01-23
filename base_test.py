import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import utils.util as util
from utils.visualizer import Visualizer
from utils import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
model = create_model(opt)
if opt.data_type == 16:
    model.half()
elif opt.data_type == 8:
    model.type(torch.uint8)
        
if opt.verbose:
    print(model)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    
    minibatch = 1 
     
    generated = model.inference(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']), \
                real_image=  Variable(data['target_rgb']))
    
    tmp = []
    tmp.extend([( 'reference1', util.tensor2im(data['reference_frames'][0,:3]))])
    if opt.num_frames >= 4:
        tmp.extend([('reference2', util.tensor2im(data['reference_frames'][0, 6:9])),
                            ('reference3', util.tensor2im(data['reference_frames'][0, 12:15])),
                            ('reference4', util.tensor2im(data['reference_frames'][0, 18:21]))])
    tmp.extend([('target_lmark', util.tensor2im(data['target_lmark'][0])),
                        ('synthesized_image', util.tensor2im(generated.data[0])),
                        ('real_image', util.tensor2im(data['target_rgb'][0]))])

    visuals = OrderedDict(tmp)
    img_path = data['v_id']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()