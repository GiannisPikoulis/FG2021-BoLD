import sys
import argparse
import collections
import os
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from logger import setup_logging
from trainer.trainer import Trainer
from skeleton_dataset import SkeletonDataset
from model.stgcn import *

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(args, config):
    
    model = Model(in_channels=args.in_channels, num_class=args.num_classes, num_dim=args.num_dimensions,
                  layout=args.layout, strategy=args.strategy, max_hop=args.max_hop, dilation=args.dilation,
                  edge_importance_weighting=args.importance_weighting)
        
    logger = config.get_logger('train') 
    
    if args.pretrained_kinetics:
        
        # 1) Load Kinetics pretrained weights.
        if not os.access('st_gcn.kinetics-6fa43f73.pth', os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/open-mmlab/mmskeleton/master/checkpoints/st_gcn.kinetics-6fa43f73.pth'
            os.system('wget ' + synset_url)
        pretrained_dict = torch.load('st_gcn.kinetics-6fa43f73.pth')
        # 2) Get current model parameter dictionary.
        model_dict = model.state_dict()
        # 3) Filter out unnecessary keys.
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 4) Overwrite entries in the existing state dictionary.
        model_dict.update(pretrained_dict) 
        # 5) Load pretrained weights in ST-GCN model.
        model.load_state_dict(model_dict)
        logger.info("Initialized ST-GCN with Kinetics pretrained weights...")
    
    logger.info("\nTotal number of network trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logger.info(model)    

    policies = model.get_optim_policies()

    train_dataset = SkeletonDataset(mode="train", normalize=args.normalize,
                                    centralize=args.centralize, random_choose=args.random_choose,
                                    random_move=args.random_move, random_shift=args.random_shift)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

    val_dataset = SkeletonDataset(mode="val", normalize=args.normalize,
                                  centralize=args.centralize, random_choose=False,
                                  random_move=False, random_shift=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    optimizer = torch.optim.SGD(policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

	# get function handles of loss and metrics
    criterion_categorical = getattr(module_loss, config['loss_categorical'])
    criterion_continuous = getattr(module_loss, config['loss_continuous'])

    metrics_categorical = [getattr(module_metric, met) for met in config['metrics_categorical']]
    metrics_continuous = [getattr(module_metric, met) for met in config['metrics_continuous']]

    trainer = Trainer(model, criterion_categorical, criterion_continuous, metrics_categorical, metrics_continuous, optimizer,
                      config=config, train_dataloader=train_loader, val_dataloader=val_loader, lr_scheduler=lr_scheduler)

    trainer.train()    
    logger.info('Best result: {}'.format(trainer.mnt_best))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch Template')
	
	# ========================= Runtime Configs ==========================
    parser.add_argument('--n_workers', default=4, type=int, help='number of data loading workers (default: %(default)s)')
    parser.add_argument('--config', default=None, type=str, help='config file path (default: %(default)s)')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: %(default)s)')
    parser.add_argument('--device', required=True, type=str, help='indices of GPUs to enable separated by commas')

	# ========================= Model Configs ==========================
    parser.add_argument('--layout', type=str, default="openpose", choices=["openpose", "ntu-rgb+d", "ntu_edge"], help="skeleton model layout (default: %(default)s)")
    parser.add_argument('--strategy', type=str, default="spatial", choices=["uniform", "distance", "spatial"], help='joint labeling strategy (default: %(default)s)')
    parser.add_argument('--in_channels', default=3, type=int, help='number of channels for joint sequences (default: %(default)s)')    
    parser.add_argument('--max_hop', default=1, type=int, help='maximum limb sequence length between neighboring joints (default: %(default)s)')
    parser.add_argument('--dilation', default=1, type=int, help='controls the spacing between the kernel points (default: %(default)s)')
    parser.add_argument('--pretrained_kinetics', default=False, action="store_true", help='load Kinetics pretrained weights (default: %(default)s)')
    parser.add_argument('--importance_weighting', default=False, action="store_true", help='apply edge importance weighting in ST-GCN units (default: %(default)s)')

	# ========================= Learning Configs ==========================
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: %(default)s)')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate (default: %(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay (default: %(default)s)')
    parser.add_argument('--num_classes', type=int, default=26, help='number of emotional classes (default: %(default)s)')
    parser.add_argument('--num_dimensions', type=int, default=3, help='number of emotional dimensions (default: %(default)s)')
    parser.add_argument('--normalize', default=False, action="store_true", help='normalize 2D joint coordinates in the range [0, 1] (default: %(default)s)')
    parser.add_argument('--centralize', default=False, action="store_true", help='normalize 2D joint coordinates in the range [-0.5, 0.5] (default: %(default)s)')
    
	# ========================= Data Augmentation Configs ==========================
    parser.add_argument('--random_move', default=False, action="store_true", help='if true, perform randomly but continuously changed transformation to input sequence (default: %(default)s)')
    parser.add_argument('--random_choose', default=False, action="store_true", help='if true, randomly choose a portion of the input sequence (default: %(default)s)')
    parser.add_argument('--random_shift', default=False, action="store_true", help='if true, randomly pad zeros at the begining or end of sequence (default: %(default)s)')
    
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
        
    if args.pretrained_kinetics and (args.strategy != 'spatial' or args.dilation != 1 or args.max_hop != 1):
        raise ValueError("in order to load Kinetics pretrained weights, the spatial labeling strategy needs to be selected with max_hop=1 and dilation=1")  
        
    main(args, config)