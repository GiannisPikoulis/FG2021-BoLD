import sys
import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from transforms import *
from logger import setup_logging
from model import loss
from trainer.trainer import Trainer
from dataset import TSNDataset
from model.models import TSN

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args, config):
    
    logger = config.get_logger('train') 
    
    model = TSN(logger=logger, num_classes=args.num_classes, num_dimensions=args.num_dimensions,
                rgb_body=args.rgb_body, rgb_context=args.rgb_context, rgb_face=args.rgb_face, 
                flow_body=args.flow_body, flow_context=args.flow_context, flow_face=args.flow_face,
                scenes=args.scenes, attributes=args.attributes, depth=(args.modality=='Depth'),
                rgbdiff_body=args.rgbdiff_body, rgbdiff_context=args.rgbdiff_context, rgbdiff_face=args.rgbdiff_face,
                arch=args.arch, consensus_type=args.consensus_type, partial_bn=args.partial_bn, embed=args.embed,
                pretrained_affectnet=args.pretrained_affectnet, pretrained_places=args.pretrained_places,
                pretrained_imagenet=args.pretrained_imagenet)
    
    logger.info("\nTotal number of network trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logger.info(model)    
            
    rgb_mean = model.rgb_mean
    rgb_std = model.rgb_std
    flow_mean = model.flow_mean
    flow_std = model.flow_std
    depth_mean = model.depth_mean
    depth_std = model.depth_std
    diff_mean = model.diff_mean
    diff_std = model.diff_std
    
    policies = model.get_optim_policies()

    rgb_normalize = GroupNormalize(rgb_mean, rgb_std)
    flow_normalize = GroupNormalize(flow_mean, flow_std)
    depth_normalize = GroupNormalize(depth_mean, depth_std)
    diff_normalize = GroupNormalize(diff_mean, diff_std)

    train_dataset = TSNDataset(mode="train", num_segments=args.train_segments,
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
                               random_shift=True,
                               context=args.context)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

    val_dataset = TSNDataset(mode="val", num_segments=args.val_segments,
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

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    optimizer = torch.optim.SGD(policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    """
    Starting epoch is set to 1
    Consider the fact that the learning rate is reduced one epoch after each milestone
    """
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

	# get function handles of loss and metrics
    criterion_categorical = getattr(module_loss, config['loss_categorical'])
    criterion_continuous = getattr(module_loss, config['loss_continuous'])

    metrics_categorical = [getattr(module_metric, met) for met in config['metrics_categorical']]
    metrics_continuous = [getattr(module_metric, met) for met in config['metrics_continuous']]

    trainer = Trainer(model, criterion_categorical, criterion_continuous, metrics_categorical, metrics_continuous, optimizer, config=config, train_dataloader=train_loader, val_dataloader=val_loader, lr_scheduler=lr_scheduler, embed=args.embed)

    trainer.train()    
    logger.info('Best result: {}'.format(trainer.mnt_best))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Multi-modal, multi-stream TSN training on the Body Language Dataset (BoLD)')
	
	# ========================= Runtime Configs ==========================
    parser.add_argument('--n_workers', default=4, type=int, help='number of data loading workers (default: %(default)s)')
    parser.add_argument('--config', default=None, type=str, help='config file path (default: %(default)s)')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: %(default)s)')
    parser.add_argument('--device', required=True, type=str, help='indices of GPUs to enable separated by commas')

	# ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet50"], help="CNN backbone architecture (default: %(default)s)")
    parser.add_argument('--train_segments', type=int, default=3, help='number of segments used during training (default: %(default)s)')
    parser.add_argument('--val_segments', type=int, default=25, help='number of segments used during validation (default: %(default)s)')
    parser.add_argument('--consensus_type', type=str, default='avg', choices=['avg', 'linear_weighting', 'attention_weighting'], help='segmental consensus function (default: %(default)s)')

	# ========================= Learning Configs ==========================
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: %(default)s)')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate (default: %(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay (default: %(default)s)')
    parser.add_argument('--partial_bn', default=False, action="store_true", help='partial batch normalization (default: %(default)s)')
    parser.add_argument('--context', default=False, action="store_true", help='load context data (default: %(default)s)')
    parser.add_argument('--embed', default=False, action="store_true", help='use embedding loss (default: %(default)s)')
    parser.add_argument('--num_classes', type=int, default=26, help='number of emotional classes (default: %(default)s)')
    parser.add_argument('--num_dimensions', type=int, default=3, help='number of emotional dimensions (default: %(default)s)')

	# ========================= Modality Config ==========================
    parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'Depth'], required=True, help='input data modality')
	
	# ========================= TSN Model Stream Configs ==========================
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
    
	# ========================= TSN Stream Pretraining Configs ==========================
    parser.add_argument('--pretrained_affectnet', default=False, action="store_true", help='load AffectNet pretrained weights, for RGB face stream (default: %(default)s)')
    parser.add_argument('--pretrained_places', default=False, action="store_true", help='load Places365 pretrained weights, for RGB context stream (default: %(default)s)')
    parser.add_argument('--pretrained_imagenet', default=False, action="store_true", help='load ImageNet pretrained weights, for RGB body stream and all Flow/RGBDiff streams (default: %(default)s)')
    
	# custom cli options to modify configuration from default values given in json file.
    custom_name = collections.namedtuple('custom_name', 'flags type target help')
    custom_epochs = collections.namedtuple('custom_epochs', 'flags type target help')
    custom_milestones = collections.namedtuple('custom_milestones', 'flags type nargs target help')
    
    options = [custom_name(['--exp_name'], type=str, target='name', help="custom experiment name (overwrites 'name' value from the configuration file"), 
               custom_epochs(['--epochs'], type=int, target='trainer;epochs', help="custom number of epochs (overwrites 'trainer->epochs' value from the configuration file"), 
               custom_milestones(['--milestones'], type=int, nargs='+', target='lr_scheduler;args;milestones', help="custom milestones for scheduler (overwrites 'lr_scheduler->args->milestones' value from the configuration file")]
    
    config = ConfigParser.from_args(parser, options)
 
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
    if args.modality == 'RGB':
        if not args.rgb_body and not args.rgb_context and not args.rgb_face:
            raise ValueError("At least one RGB stream needs to be specified when using the RGB input modality")  
        if (args.scenes or args.attributes) and not args.rgb_context:
            raise ValueError("The scenes and attributes streams require the RGB context stream")  
        if args.context != args.rgb_context:
            raise ValueError("The RGB context stream requires context data to be loaded from the dataset")  
    elif args.modality == 'Flow':
        if not args.flow_body and not args.flow_context and not args.flow_face:
            raise ValueError("At least one Optical Flow stream needs to be specified when using the Optical Flow input modality")
        if args.context != args.flow_context:
            raise ValueError("The Optical Flow context stream requires context data to be loaded from the dataset")  
    elif args.modality == 'RGBDiff':
        if not args.rgbdiff_body and not args.rgbdiff_context and not args.rgbdiff_face:
            raise ValueError("At least one RGB Difference stream needs to be specified when using the RGB Difference input modality")
        if args.context != args.rgbdiff_context:
            raise ValueError("The RGB Difference context stream requires context data to be loaded from the dataset")               
    main(args, config)