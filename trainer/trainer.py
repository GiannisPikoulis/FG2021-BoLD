import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, make_barplot
import matplotlib as mpl
import random
import torch.nn.functional as F

mpl.use('Agg')
import matplotlib.pyplot as plt
import model.metric
import model.loss


class Trainer(BaseTrainer):
    
    def __init__(self, model, criterion_categorical, criterion_continuous, metric_categorical, metric_continuous, optimizer, 
                 config, train_dataloader, val_dataloader, lr_scheduler=None, len_epoch=None, embed=False):
        
        super().__init__(model, optimizer, config)
        
        if model.num_rgb_mods > 0:
            self.rgb_body = model.rgb_body
            self.rgb_context = model.rgb_context
            self.rgb_face = model.rgb_face
            self.mod = 'RGB'
        elif model.num_flow_mods > 0:
            self.flow_body = model.flow_body
            self.flow_context = model.flow_context
            self.flow_face = model.flow_face
            self.mod = 'Flow'
        elif model.num_diff_mods > 0:
            self.rgbdiff_body = model.rgbdiff_body
            self.rgbdiff_context = model.rgbdiff_context
            self.rgbdiff_face = model.rgbdiff_face
            self.mod = 'RGBDiff'
        elif model.num_depth_mods > 0:
            self.mod = 'Depth'
        else:
            self.mod = 'Skeleton'
            
        self.config = config    
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.do_validation = self.val_dataloader is not None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.embed = embed                     
        
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(train_dataloader)
            self.len_epoch = len_epoch
        
        self.log_step = int(self.len_epoch / 10)    

        self.metric_categorical = metric_categorical
        self.metric_continuous = metric_continuous

        self.criterion_continuous = criterion_continuous
        self.criterion_categorical = criterion_categorical

        self.categorical_class_metrics = [_class + "_" + m.__name__ for _class in val_dataloader.dataset.categorical_emotions for m in self.metric_categorical]

        self.continuous_class_metrics = [_class + "_" + m.__name__ for _class in val_dataloader.dataset.continuous_emotions for m in self.metric_continuous]

        self.train_metrics = MetricTracker('Loss', 'Categorical Loss', 'Continuous Loss', 'Embedding Loss', 'mAP', 'mRA', 'mR2', 'mSE', 'ERS', writer=self.writer)
        self.val_metrics = MetricTracker('Loss', 'Categorical Loss', 'Continuous Loss', 'Embedding Loss', 'mAP', 'mRA', 'mR2', 'mSE', 'ERS', writer=self.writer)     


    def _train_epoch(self, epoch, phase="train"):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metrics in this epoch.
        """
        
        if phase == "train": 
            self.logger.info("Starting training phase for epoch: {}".format(epoch)) 
            self.logger.info("Printing learning rates...")
            for param_group in self.optimizer.param_groups:
                self.logger.info(param_group['lr'])       
            self.model.train()
            self.train_metrics.reset()
            torch.set_grad_enabled(True)
            metrics = self.train_metrics
        
        elif phase == "val":
            self.logger.info("Starting validation phase for epoch: {}".format(epoch))
            self.model.eval()
            self.val_metrics.reset()
            torch.set_grad_enabled(False)            
            metrics = self.val_metrics

        _outputs_categorical = []
        _outputs_continuous = []
        _targets_categorical = []
        _targets_continuous = []
        
        running_loss = 0
        running_cat_loss = 0
        running_cont_loss = 0
        running_embed_loss = 0
        
        total_loss = 0
        total_cat_loss = 0
        total_cont_loss = 0
        total_embed_loss = 0
        
        dataloader = self.train_dataloader if phase == "train" else self.val_dataloader
        
        for batch_idx, batch_data in enumerate(dataloader):
            
            inputs = {}
            if self.mod == 'RGB':
                if self.rgb_body:
                    inputs['body'] = batch_data[0].to(self.device) 
                if self.rgb_face:
                    inputs['face'] = batch_data[1].to(self.device) 
                if self.rgb_context:
                    inputs['context'] = batch_data[6].to(self.device)                     
            elif self.mod == 'Flow':
                if self.flow_body:
                    inputs['body'] = batch_data[0].to(self.device) 
                if self.flow_face:
                    inputs['face'] = batch_data[1].to(self.device)  
                if self.flow_context:
                    inputs['context'] = batch_data[6].to(self.device)                           
            elif self.mod == 'RGBDiff':
                if self.rgbdiff_body:
                    inputs['body'] = batch_data[0].to(self.device) 
                if self.rgbdiff_face:
                    inputs['face'] = batch_data[1].to(self.device)    
                if self.rgbdiff_context:
                    inputs['context'] = batch_data[6].to(self.device)               
            elif self.mod == 'Depth':
                raise NotImplementedError()
            elif self.mod == 'Skeleton':
                inputs['skeleton'] = batch_data[0].to(self.device) 
 
            if self.mod != 'Skeleton':
                embeddings = batch_data[2].to(self.device)      
                target_categorical = batch_data[3].to(self.device) 
                target_continuous= batch_data[4].to(self.device)
            else:
                target_categorical = batch_data[1].to(self.device) 
                target_continuous= batch_data[2].to(self.device)
            
            if phase == "train":
                self.optimizer.zero_grad()
                
            if self.mod != 'Skeleton':
                out = self.model(inputs, dataloader.dataset.num_segments)                
            else:
                out = self.model(inputs)
                
            loss = 0
            
            loss_categorical = self.criterion_categorical(out['categorical'], target_categorical)
            loss += loss_categorical
            running_cat_loss += loss_categorical.item()
            
            loss_continuous = self.criterion_continuous(torch.sigmoid(out['continuous']), target_continuous)
            loss += loss_continuous
            running_cont_loss += loss_continuous.item()
            
            if self.embed:
                loss_embed = model.loss.mse_center_loss(out['embeddings'], embeddings, target_categorical)
                loss += loss_embed
                running_embed_loss += loss_embed.item()
                total_embed_loss += loss_embed.item()
                
            if phase == "train":
                loss.backward()
                self.optimizer.step() 
                
            running_loss += loss.item()    
            total_loss += loss.item()
            total_cat_loss += loss_categorical.item()
            total_cont_loss += loss_continuous.item()
            
            output_categorical = out['categorical'].cpu().detach().numpy()
            targ_categorical = target_categorical.cpu().detach().numpy()
            _outputs_categorical.append(output_categorical)
            _targets_categorical.append(targ_categorical)
        
            output_continuous = torch.sigmoid(out['continuous']).cpu().detach().numpy()
            targ_continuous = target_continuous.cpu().detach().numpy()
            _outputs_continuous.append(output_continuous)
            _targets_continuous.append(targ_continuous)  

            if (batch_idx % self.log_step == self.log_step - 1) and phase == 'train':
                self.logger.info('[Epoch: {}] {} [Total Loss: {:.4f}] [Categorical Loss: {:.4f}] [Continuous Loss: {:.4f}] [Embedding Loss: {:.4f}]'.format(epoch,
                                 self._progress(batch_idx),
                                 running_loss / self.log_step, running_cat_loss / self.log_step, running_cont_loss / self.log_step, running_embed_loss / self.log_step))

                running_loss = 0
                running_cat_loss = 0
                running_cont_loss = 0
                running_embed_loss = 0

            if batch_idx == self.len_epoch and phase == 'train':
                break
            
        self.writer.set_step(epoch, phase)

        if phase == 'val':
            metrics.update('Loss', total_loss / len(dataloader))
            metrics.update('Categorical Loss', total_cat_loss / len(dataloader))
            metrics.update('Continuous Loss', total_cont_loss / len(dataloader))
            if self.embed:
                metrics.update('Embedding Loss', total_embed_loss / len(dataloader))
        else:
            metrics.update('Loss', total_loss / self.len_epoch)
            metrics.update('Categorical Loss', total_cat_loss / self.len_epoch)
            metrics.update('Continuous Loss', total_cont_loss / self.len_epoch)
            if self.embed:
                metrics.update('Embedding Loss', total_embed_loss / self.len_epoch)  
            
        out_cat = np.vstack(_outputs_categorical)
        target_cat = np.vstack(_targets_categorical)
        
        target_cat[target_cat >= 0.5] = 1
        target_cat[target_cat < 0.5] = 0

        _ap = model.metric.average_precision(out_cat, target_cat)
        _ra = model.metric.roc_auc(out_cat, target_cat)
        metrics.update("mAP", np.mean(_ap))
        metrics.update("mRA", np.mean(_ra))

        out_cont = np.vstack(_outputs_continuous)
        target_cont = np.vstack(_targets_continuous)

        mse = model.metric.mean_squared_error(out_cont, target_cont)
        _r2 = model.metric.r2(out_cont, target_cont)
        metrics.update("mR2", np.mean(_r2))
        metrics.update("mSE", np.mean(mse))
        metrics.update("ERS", model.metric.ERS(np.mean(_r2), np.mean(_ap), np.mean(_ra)))
        
        log = metrics.result()
        
        self.writer.add_figure('%s AP per class' % phase, make_barplot(_ap, self.val_dataloader.dataset.categorical_emotions, 'average precision'))
        self.writer.add_figure('%s ROC AUC per class' % phase, make_barplot(_ra, self.val_dataloader.dataset.categorical_emotions, 'roc auc'))
        self.writer.add_figure('%s R2 per dimension' % phase, make_barplot(_r2, self.val_dataloader.dataset.continuous_emotions, 'r2'))

        if phase == "train":
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.do_validation:
                val_log = self._train_epoch(epoch, phase="val")
                log.update(**{'Validation ' + k: v for k, v in val_log.items()})
            
            return log

        elif phase == "val":
            self.writer.save_results(out_cat, "out_cat")
            self.writer.save_results(out_cont, "out_cont")
            return metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_dataloader, 'n_samples'):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)