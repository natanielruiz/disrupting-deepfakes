import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import torch.nn.functional as F

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
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

# Initialize Metrics
l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
n_dist, n_samples = 0, 0
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 

    print(i)

    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:
        # Get ground-truth output
        with torch.no_grad():
            generated_noattack = model.inference(data['label'], data['inst'], data['image'])
        # Attack
        adv_image, perturb = model.attack(data['label'], data['inst'], data['image'], target=generated_noattack)
        # Get output from adversarial sample
        with torch.no_grad():
            generated, adv_img = model.inference_attack(data['label'], data['inst'], data['image'], perturb)
        
    visuals = OrderedDict([('original_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('input_label', util.tensor2label(adv_img.data[0], opt.label_nc)),
                           ('attacked_image', util.tensor2im(generated.data[0])),
                           ('noattack', util.tensor2im(generated_noattack.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

    # Compute metrics
    l1_error += F.l1_loss(generated, generated_noattack)
    l2_error += F.mse_loss(generated, generated_noattack)
    l0_error += (generated - generated_noattack).norm(0)
    min_dist += (generated - generated_noattack).norm(float('-inf'))
    if F.mse_loss(generated, generated_noattack) > 0.05:
        n_dist += 1
    n_samples += 1

    generated, genereated_noattack, adv_image, adv_img, perturb = None, None, None, None, None
    data = None

# Print metrics
print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, 
l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

webpage.save()
