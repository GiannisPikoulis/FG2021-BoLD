import argparse
import time
import os
import sys
import numpy as np
import torchvision
from tools.tools import *
import model.metric as module_metric
from skeleton_dataset import SkeletonDataset
from model.stgcn import *
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


# options
parser = argparse.ArgumentParser(description="Run inference on BoLD with ST-GCN")

parser.add_argument('--num_classes', type=int, default=26, help='number of emotional classes (default: %(default)s)')
parser.add_argument('--num_dimensions', type=int, default=3, help='number of emotional dimensions (default: %(default)s)')

parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: %(default)s)')
parser.add_argument('--in_channels', default=3, type=int, help='number of channels for joint sequences (default: %(default)s)')    
parser.add_argument('--layout', type=str, default="openpose", choices=["openpose", "ntu-rgb+d", "ntu_edge"], help="skeleton model layout (default: %(default)s)")
parser.add_argument('--strategy', type=str, default="spatial", choices=["uniform", "distance", "spatial"], help='joint labeling strategy (default: %(default)s)')
parser.add_argument('--max_hop', default=1, type=int, help='maximum limb sequence length between neighboring joints (default: %(default)s)')
parser.add_argument('--dilation', default=1, type=int, help='controls the spacing between the kernel points (default: %(default)s)')
parser.add_argument('--importance_weighting', default=False, action="store_true", help='apply edge importance weighting in ST-GCN units (default: %(default)s)')
parser.add_argument('--normalize', default=False, action="store_true", help='normalize 2D joint coordinates in the range [0, 1] (default: %(default)s)')
parser.add_argument('--centralize', default=False, action="store_true", help='normalize 2D joint coordinates in the range [-0.5, 0.5] (default: %(default)s)')

parser.add_argument('--checkpoint', required=True, type=str, help='pretrained model checkpoint')
parser.add_argument('--output_dir', required=True, type=str, help='directory where to store outputs')
parser.add_argument('--exp_name', type=str, required=True, help='custom experiment name')

parser.add_argument('--n_workers', default=4, type=int, help='number of data loading workers (default: %(default)s)')
parser.add_argument('--device', required=True, type=str, help='indices of GPUs to enable separated by commas')
parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use (default: %(default)s)')
parser.add_argument('--mode', required=True, type=str, choices=['val', 'test'], help='type of inference to run (default: %(default)s)')
parser.add_argument('--save_outputs', default=False, action="store_true", help='whether to save outputs produced during inference (default: %(default)s)')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device    

model = Model(in_channels=args.in_channels, num_class=args.num_classes, num_dim=args.num_dimensions,
              layout=args.layout, strategy=args.strategy, max_hop=args.max_hop, dilation=args.dilation,
              edge_importance_weighting=args.importance_weighting)
            
_outputs_categorical = []
_outputs_continuous = []
_targets_categorical = []
_targets_continuous = []
               
dataset = SkeletonDataset(mode=args.mode, normalize=args.normalize,
                          centralize=args.centralize, random_choose=False,
                          random_move=False, random_shift=False)

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
            inputs['skeleton'] = batch_data[0].to(device) 
            target_categorical = batch_data[1].to(device) 
            target_continuous= batch_data[2].to(device)          
        else:
            inputs['skeleton'] = batch_data.to(device) 
            
        out = model(inputs)
            
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
        np.savetxt(os.path.join(args.output_dir, args.exp_name, args.mode, 'output.csv'), combined, delimiter=",", fmt='%1.6f')   
        print('Done saving {} outputs!'.format(args.mode))