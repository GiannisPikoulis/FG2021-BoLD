import argparse
import time
import os
import sys
import numpy as np
import torchvision
from transforms import *
import model.metric as module_metric
from dataset import TSNDataset
from model.models import TSN
from utils import MetricTracker
from collections import OrderedDict

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def _prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids    

if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser(description="Run inference on BoLD with TSN")

    parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'Depth'], required=True, help='input data modality')
    parser.add_argument('--num_classes', type=int, default=26, help='number of emotional classes (default: %(default)s)')
    parser.add_argument('--num_dimensions', type=int, default=3, help='number of emotional dimensions (default: %(default)s)')
    parser.add_argument('--num_segments', type=int, default=25, help='number of segments to use during inference (default: %(default)s)')
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet50"], help="CNN backbone architecture (default: %(default)s)")
    parser.add_argument('--consensus_type', type=str, default='avg', choices=['avg', 'linear_weighting', 'attention_weighting'], help='segmental consensus function (default: %(default)s)')
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: %(default)s)')

    parser.add_argument('--rgb_body', default=False, action="store_true", help='use RGB body stream (default: %(default)s)')
    parser.add_argument('--rgb_context', default=False, action="store_true", help='use RGB context stream (default: %(default)s)')
    parser.add_argument('--rgb_face', default=False, action="store_true", help='use RGB face stream (default: %(default)s)')
    parser.add_argument('--scenes', default=False, action="store_true", help='use RGB scenes stream (default: %(default)s)')
    parser.add_argument('--attributes', default=False, action="store_true", help='use RGB attributes stream (default: %(default)s)')
        
    parser.add_argument('--flow_body', default=False, action="store_true", help='use Flow body stream (default: %(default)s)')
    parser.add_argument('--flow_context', default=False, action="store_true", help='use Flow context stream (default: %(default)s)')
    parser.add_argument('--flow_face', default=False, action="store_true", help='use Flow face stream (default: %(default)s)')

    parser.add_argument('--rgbdiff_body', default=False, action="store_true", help='use RGBDiff body stream (default: %(default)s)')
    parser.add_argument('--rgbdiff_context', default=False, action="store_true", help='use RGBDiff context stream (default: %(default)s)')
    parser.add_argument('--rgbdiff_face', default=False, action="store_true", help='use RGBDiff face stream (default: %(default)s)')

    parser.add_argument('--partial_bn', default=False, action="store_true", help='partial batch normalization (default: %(default)s)')
    parser.add_argument('--context', default=False, action="store_true", help='load context data (default: %(default)s)')
    parser.add_argument('--embed', default=False, action="store_true", help='use embedding loss (default: %(default)s)')

    parser.add_argument('--checkpoint', required=True, type=str, help='pretrained model checkpoint')
    parser.add_argument('--output_dir', required=True, type=str, help='directory where to store outputs')
    parser.add_argument('--exp_name', type=str, required=True, help='custom experiment name')

    parser.add_argument('--n_workers', default=4, type=int, help='number of data loading workers (default: %(default)s)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable separated by commas (default: all)')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use (default: %(default)s)')
    parser.add_argument('--mode', required=True, type=str, choices=['val', 'test'], help='type of inference to run (default: %(default)s)')
    parser.add_argument('--save_outputs', default=False, action="store_true", help='whether to save outputs produced during inference (default: %(default)s)')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device    

    if args.modality == 'RGB':
        if not args.rgb_body and not args.rgb_context and not args.rgb_face:
            raise ValueError("At least one RGB stream needs to be specified when using the RGB input modality")  
        if (args.scenes or args.attributes) and not args.rgb_context:
            raise ValueError("The 'scenes' and 'attributes' streams require the RGB context stream")  
        if args.context != args.rgb_context:
            raise ValueError("The RGB 'context' stream requires 'context' data to be loaded from the dataset")  
    elif args.modality == 'Flow':
        if not args.flow_body and not args.flow_context and not args.flow_face:
            raise ValueError("At least one Optical Flow stream needs to be specified when using the Optical Flow input modality")
        if args.context != args.flow_context:
            raise ValueError("The Optical Flow 'context' stream requires 'context' data to be loaded from the dataset")  
    elif args.modality == 'RGBDiff':
        if not args.rgbdiff_body and not args.rgbdiff_context and not args.rgbdiff_face:
            raise ValueError("At least one RGB Difference stream needs to be specified when using the RGB Difference input modality")  	
        if args.context != args.rgbdiff_context:
            raise ValueError("The RGB Difference 'context' stream requires 'context' data to be loaded from the dataset")   

    model = TSN(logger=None, num_classes=args.num_classes, num_dimensions=args.num_dimensions,
                rgb_body=args.rgb_body, rgb_context=args.rgb_context, rgb_face=args.rgb_face, 
                flow_body=args.flow_body, flow_context=args.flow_context, flow_face=args.flow_face,
                scenes=args.scenes, attributes=args.attributes, depth=(args.modality=='Depth'),
                rgbdiff_body=args.rgbdiff_body, rgbdiff_context=args.rgbdiff_context, rgbdiff_face=args.rgbdiff_face,
                arch=args.arch, consensus_type=args.consensus_type, partial_bn=args.partial_bn, embed=args.embed,
                pretrained_affectnet=False, pretrained_places=False, pretrained_imagenet=False)

    _outputs_categorical = []
    _outputs_continuous = []
    _targets_categorical = []
    _targets_continuous = []
        
    rgb_mean = model.rgb_mean
    rgb_std = model.rgb_std
    flow_mean = model.flow_mean
    flow_std = model.flow_std
    depth_mean = model.depth_mean
    depth_std = model.depth_std
    diff_mean = model.diff_mean
    diff_std = model.diff_std    
        
    rgb_normalize = GroupNormalize(rgb_mean, rgb_std)
    flow_normalize = GroupNormalize(flow_mean, flow_std)    
    depth_normalize = GroupNormalize(depth_mean, depth_std)
    diff_normalize = GroupNormalize(diff_mean, diff_std)
                
    dataset = TSNDataset(mode=args.mode, num_segments=args.num_segments,
                        inp_type=args.modality,
                        rgb_transform=torchvision.transforms.Compose([
                            GroupScale((224,224)),
                            Stack(roll=False),
                            ToTorchFormatTensor(div=True),
                            rgb_normalize
                            ]),
                        flow_transform=torchvision.transforms.Compose([
                            GroupScale((224,224)),
                            Stack(roll=False),
                            ToTorchFormatTensor(div=True),
                            flow_normalize
                            ]),
                        depth_transform=torchvision.transforms.Compose([
                            GroupScale((224,224)),
                            Stack(roll=False),
                            ToTorchFormatTensor(div=True),
                            depth_normalize
                            ]),            
                        diff_transform=torchvision.transforms.Compose([
                            GroupScale((224,224)),
                            Stack(roll=False),
                            ToTorchFormatTensor(div=True),
                            diff_normalize
                            ]),                                
                        random_shift=False,
                        context=args.context)

    print('\nSet: {}'.format(args.mode))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    metrics = MetricTracker('ERS', 'mAP', 'mRA', 'mR2', 'mSE', writer=None)

    # Create directory to save predictions
    if not os.path.exists(os.path.join(args.output_dir, args.exp_name, args.mode)):
        os.makedirs(os.path.join(args.output_dir, args.exp_name, args.mode)) 

    # Load checkpoint    
    print('Checkpoint path: {}'.format(args.checkpoint))     
    checkpoint = torch.load(args.checkpoint)
                    
    new_state_dict = OrderedDict()

    for k, v in checkpoint['state_dict'].items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
                
    model.load_state_dict(new_state_dict)

    # setup GPU device if available, move model into configured device
    device, device_ids = _prepare_device(n_gpu_use=args.n_gpu)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
            
    print("Total number of network trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        
    model.eval()
                
    with torch.set_grad_enabled(False):
        
        for batch_idx, batch_data in enumerate(dataloader):
            
            inputs = {}
            if args.mode != 'test':    
                if args.modality == 'RGB':
                    if model.rgb_body:
                        inputs['body'] = batch_data[0].to(device) 
                    if model.rgb_face:
                        inputs['face'] = batch_data[1].to(device) 
                    if model.rgb_context:
                        inputs['context'] = batch_data[6].to(device)                     
                elif args.modality == 'Flow':
                    if model.flow_body:
                        inputs['body'] = batch_data[0].to(device) 
                    if model.flow_face:
                        inputs['face'] = batch_data[1].to(device)  
                    if model.flow_context:
                        inputs['context'] = batch_data[6].to(device)                           
                elif args.modality == 'RGBDiff':
                    if args.model.rgbdiff_body:
                        inputs['body'] = batch_data[0].to(device) 
                    if args.model.rgbdiff_face:
                        inputs['face'] = batch_data[1].to(device)    
                    if args.model.rgbdiff_context:
                        inputs['context'] = batch_data[6].to(device)               
                else:
                    raise NotImplementedError()  
                
                embeddings = batch_data[2].to(device)      
                target_categorical = batch_data[3].to(device) 
                target_continuous= batch_data[4].to(device)  
            
            else:    
                if args.modality == 'RGB':
                    if model.rgb_body:
                        inputs['body'] = batch_data[0].to(device) 
                    if model.rgb_face:
                        inputs['face'] = batch_data[1].to(device) 
                    if model.rgb_context:
                        inputs['context'] = batch_data[4].to(device)                     
                elif args.modality == 'Flow':
                    if model.flow_body:
                        inputs['body'] = batch_data[0].to(device) 
                    if model.flow_face:
                        inputs['face'] = batch_data[1].to(device)  
                    if model.flow_context:
                        inputs['context'] = batch_data[4].to(device)                           
                elif args.modality == 'RGBDiff':
                    if args.model.rgbdiff_body:
                        inputs['body'] = batch_data[0].to(device) 
                    if args.model.rgbdiff_face:
                        inputs['face'] = batch_data[1].to(device)    
                    if args.model.rgbdiff_context:
                        inputs['context'] = batch_data[4].to(device)               
                else:
                    raise NotImplementedError()  
                
                embeddings = batch_data[2].to(device)               
                
            out = model(inputs, args.num_segments)
                
            output_categorical = out['categorical'].cpu().detach().numpy()
            _outputs_categorical.append(output_categorical)            
            if args.mode != 'test':
                targ_categorical = target_categorical.cpu().detach().numpy()
                _targets_categorical.append(targ_categorical)                
            
            output_continuous = torch.sigmoid(out['continuous']).cpu().detach().numpy()
            _outputs_continuous.append(output_continuous)            
            if args.mode != 'test':            
                targ_continuous = target_continuous.cpu().detach().numpy()
                _targets_continuous.append(targ_continuous)

    out_cat = np.vstack(_outputs_categorical)    
    if args.mode != 'test':    
        target_cat = np.vstack(_targets_categorical)
        
    if args.mode != 'test':    
        target_cat[target_cat >= 0.5] = 1
        target_cat[target_cat < 0.5] = 0
        _ap = module_metric.average_precision(out_cat, target_cat)
        _ra = module_metric.roc_auc(out_cat, target_cat)
        metrics.update("mAP", np.mean(_ap))
        metrics.update("mRA", np.mean(_ra))

    out_cont = np.vstack(_outputs_continuous)    
    if args.mode != 'test':        
        target_cont = np.vstack(_targets_continuous)
        
    if args.mode != 'test':    
        mse = module_metric.mean_squared_error(out_cont, target_cont)
        _r2 = module_metric.r2(out_cont, target_cont)
        metrics.update("mR2", np.mean(_r2))
        metrics.update("mSE", np.mean(mse))
        metrics.update("ERS", module_metric.ERS(np.mean(_r2), np.mean(_ap), np.mean(_ra)))
        
    if args.mode != 'test':    
        log = metrics.result()
        print('Printing {} performance metrics...'.format(args.mode)) 
        print(log)
        
    if args.mode != 'test':
        if args.save_outputs:
            np.save(os.path.join(args.output_dir, args.exp_name, args.mode, 'output_cat.npy'), out_cat)
            np.save(os.path.join(args.output_dir, args.exp_name, args.mode, 'output_cont.npy'), out_cont)
            np.save(os.path.join(args.output_dir, args.exp_name, args.mode, 'target_cat.npy'), target_cat)
            np.save(os.path.join(args.output_dir, args.exp_name, args.mode, 'target_cont.npy'), target_cont)
            print('Done saving {} outputs and targets!'.format(args.mode))
    else:
        combined = np.hstack((out_cont, out_cat))
        if args.save_outputs:
            np.save(os.path.join(args.output_dir, args.exp_name, args.mode, 'output_cat.npy'), out_cat)
            np.save(os.path.join(args.output_dir, args.exp_name, args.mode, 'output_cont.npy'), out_cont)        
            np.savetxt(os.path.join(args.output_dir, args.exp_name, args.mode, 'output.csv'), combined, delimiter=",", fmt='%1.6f') 
            print('Done saving {} outputs!'.format(args.mode))