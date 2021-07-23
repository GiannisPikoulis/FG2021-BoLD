from torch import nn
import torch.nn.functional
import torch
from .ops.basic_ops import ConsensusModule, Identity
from .places365_SUN_model import *
from torch.nn.init import normal, constant
from torch.nn import Parameter
import torchvision
import numpy as np


class Face_Model_18(nn.Module):
    
    def __init__(self):
        super(Face_Model_18, self).__init__()
        """
        --> ResNet18 is pre-trained on ImageNet
        """
        resnet18 = torchvision.models.resnet18(pretrained = True)
        self.resnet18_face = nn.Sequential(*list(resnet18.children())[:-1])
        self.categorical_layer = nn.Linear(in_features = 512, out_features = 8)
        self.continuous_layer = nn.Linear(in_features = 512, out_features = 2)

    def forward(self, face):
        x = self.resnet18_face(face).squeeze()
        out_cat = self.categorical_layer(x)
        out_cont = self.continuous_layer(x)
        return out_cat, out_cont

    
class Face_Model_50(nn.Module):
    
    def __init__(self):
        super(Face_Model_50, self).__init__()
        """
        --> ResNet18 is pre-trained on ImageNet
        """
        resnet50 = torchvision.models.resnet50(pretrained = True)
        self.resnet50_face = nn.Sequential(*list(resnet50.children())[:-1])
        self.categorical_layer = nn.Linear(in_features = 2048, out_features = 8)
        self.continuous_layer = nn.Linear(in_features = 2048, out_features = 2)
 

    def forward(self, face):
        x = self.resnet50_face(face).squeeze()
        out_cat = self.categorical_layer(x)
        out_cont = self.continuous_layer(x)
        return out_cat, out_cont    

    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

            
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'    


class TSN(nn.Module):
    
    def __init__(self, logger, num_classes, num_dimensions, pretrained_affectnet, pretrained_places, pretrained_imagenet,
                 rgb_body, rgb_context, rgb_face, 
                 flow_body, flow_context, flow_face,
                 scenes, attributes, depth, 
                 rgbdiff_body, rgbdiff_context, rgbdiff_face,
                 arch='resnet18', consensus_type='avg', partial_bn=False, embed=False):
        
        super(TSN, self).__init__()
        self.rgb_length = 1
        self.flow_length = 5
        self.depth_length = 5
        self.diff_length = 5
        
        self.num_classes = num_classes
        self.num_dimensions = num_dimensions
        
        self.rgb_context = rgb_context
        self.rgb_body = rgb_body
        self.rgb_face = rgb_face
        self.flow_context = flow_context
        self.flow_body = flow_body
        self.flow_face = flow_face
        self.depth = depth
        self.rgbdiff_body = rgbdiff_body
        self.rgbdiff_context = rgbdiff_context
        self.rgbdiff_face = rgbdiff_face
        
        self.scenes = scenes
        self.attributes = attributes
        
        self.pretrained_affectnet = pretrained_affectnet
        self.pretrained_places = pretrained_places
        self.pretrained_imagenet = pretrained_imagenet
        
        self.extra_feats = 0
        self.logger = logger
        self.modalities = list()

        if self.rgb_context:
            self.modalities.append("RGB Context")
        if self.rgb_body:
            self.modalities.append("RGB Body")
        if self.rgb_face:
            self.modalities.append("RGB Face")
        if self.flow_body:
            self.modalities.append("Flow Body")
        if self.flow_context:
            self.modalities.append("Flow Context")
        if self.flow_face:
            self.modalities.append("Flow Face")        
        if self.scenes:
            self.modalities.append("Places365 Scene Scores")
            self.extra_feats += 365
        if self.attributes:
            self.modalities.append("SUN Attribute Scores")
            self.extra_feats += 102
        if self.depth:
            raise NotImplementedError
        if self.rgbdiff_body:
            self.modalities.append("RGBDiff Body")
        if self.rgbdiff_context:
            self.modalities.append("RGBDiff Context")
        if self.rgbdiff_face:
            self.modalities.append("RGBDiff Face")
             
        self.consensus_type = consensus_type
        self.embed = embed
        self._enable_pbn = partial_bn
        self.arch = arch
        
        if self.arch == 'resnet18' or self.arch == 'resnet34':
            self.num_feats = 512
        else:
            self.num_feats = 2048
        
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        self.flow_mean = [0.5]
        self.flow_std = [np.mean(self.rgb_std)]
        
        self.depth_mean = [0.5]*3
        self.depth_std = [np.mean(self.rgb_std)]*3
        
        self.diff_mean = [0]*3 
        self.diff_std = [np.mean(self.rgb_std)*2]*3  
        
        if self.logger:
            self.logger.info(("""
            Initializing TSN with the following configuration:
            CNN Backbones:      {}
            Input Modalities:   {}
            Consensus Module:   {}
            Embedding Loss:     {}
            Partial BN:         {}
            """.format(self.arch, self.modalities, self.consensus_type, self.embed, self._enable_pbn)))
      
        self.num_rgb_mods = 0
        self.num_flow_mods = 0
        self.num_depth_mods = 0
        self.num_diff_mods = 0
        
        """
        Construct Models
        """
        self.abbr = ''
        if self.rgb_body:
            self.rgb_body_model = self._prepare_body_model(arch=self.arch, pretrained=self.pretrained_imagenet)
            self.num_rgb_mods += 1
            self.abbr += 'b'
            
        if self.rgb_context:
            self.rgb_context_model = self._prepare_context_model(arch=self.arch, pretrained=self.pretrained_places)
            self.num_rgb_mods += 1
            self.abbr += 'c'
        
        if self.rgb_face:
            self.rgb_face_model = self._prepare_face_model(arch=self.arch, pretrained=self.pretrained_affectnet)
            self.num_rgb_mods += 1    
            self.abbr += 'f'
            
        if self.flow_body:
            self.flow_body_model = self._construct_flow_model(arch=self.arch, mode='body', pretrained=self.pretrained_imagenet)
            self.num_flow_mods += 1
            self.abbr += 'b'
        
        if self.flow_context:
            self.flow_context_model = self._construct_flow_model(arch=self.arch, mode='context', pretrained=self.pretrained_imagenet)
            self.num_flow_mods += 1
            self.abbr += 'c'
        
        if self.flow_face:
            self.flow_face_model = self._construct_flow_model(arch=self.arch, mode='face', pretrained=self.pretrained_imagenet)
            self.num_flow_mods += 1
            self.abbr += 'f'
        
        if self.scenes or self.attributes:
            _, _, _, W_attribute = load_labels()
            self.register_buffer('W_attribute', torch.tensor(W_attribute).unsqueeze(0))
            self.W_attribute.requires_grad = False
            self.unified_places_model = load_model()
            
            if self.scenes:
                self.abbr += 's'
            if self.attributes:
                self.abbr += 'a'
            
            for param in self.unified_places_model.parameters():
                param.requires_grad = False     
                
            self.features_blobs = {}
            features_names = ['avgpool'] # this is the avg. pooling layer of the wideresnet
        
            for name in features_names:
                self.unified_places_model._modules.get(name).register_forward_hook(self.hook_feature)        
        
        if self.depth:
            self.depth_model = self._construct_depth_model(arch=self.arch, pretrained=self.pretrained_imagenet)
            self.num_depth_mods += 1
        
        if self.rgbdiff_body:
            self.rgbdiff_body_model = self._construct_diff_model(arch=self.arch, mode='body', pretrained=self.pretrained_imagenet)
            self.num_diff_mods += 1
            self.abbr += 'b'
        
        if self.rgbdiff_context:
            self.rgbdiff_context_model = self._construct_diff_model(arch=self.arch, mode='context', pretrained=self.pretrained_imagenet)
            self.num_diff_mods += 1        
            self.abbr += 'c'
          
        if self.rgbdiff_face:
            self.rgbdiff_face_model = self._construct_diff_model(arch=self.arch, mode='face', pretrained=self.pretrained_imagenet)
            self.num_diff_mods += 1        
            self.abbr += 'f'
        
        self._prepare_tsn()

        if self.consensus_type=="avg":
            self.consensus_cat = ConsensusModule(self.consensus_type)
            self.consensus_cont = ConsensusModule(self.consensus_type)
            if self.logger:
                self.logger.info("Consensus type: {}".format(self.consensus_type))
        elif self.consensus_type=="linear_weighting":
            self.linear_weighting_cat = nn.Linear(in_features=7, out_features=1, bias=False)
            self.linear_weighting_cont = nn.Linear(in_features=7, out_features=1, bias=False)
            if self.logger:
                self.logger.info("Consensus type: {}".format(self.consensus_type))
        elif self.consensus_type=="attention_weighting":
            if self.num_rgb_mods > 0:
                self.attention_weighting_cat = nn.Linear(in_features=self.num_rgb_mods*self.num_feats+self.extra_feats,
                                                         out_features=1, bias=False)
                self.attention_weighting_cont = nn.Linear(in_features=self.num_rgb_mods*self.num_feats+self.extra_feats,
                                                          out_features=1, bias=False)            
            elif self.num_flow_mods > 0:
                self.attention_weighting_cat = nn.Linear(in_features=self.num_flow_mods*self.num_feats,
                                                         out_features=1, bias=False)                
                self.attention_weighting_cont = nn.Linear(in_features=self.num_flow_mods*self.num_feats,
                                                          out_features=1, bias=False)               
            elif self.num_rgbdiff_mods > 0:
                self.attention_weighting = nn.Linear(in_features=self.num_rgbdiff_mods*self.num_feats,
                                                     out_features=1, bias=False)                 
        if self.embed:
            if self.consensus_type=='avg':
                self.consensus_embed = ConsensusModule(self.consensus_type)
            elif self.consensus_type=='linear_weighting':
                self.linear_weighting_embed = nn.Linear(in_features=7, out_features=1, bias=False)
            elif self.consensus_type=='attention_weighting':
                if self.num_rgb_mods > 0:
                    self.attention_weighting_embed = nn.Linear(in_features=self.num_rgb_mods*self.num_feats+self.extra_feats,
                                                               out_features=1, bias=False)
                elif self.num_flow_mods > 0:
                    self.attention_weighting_embed = nn.Linear(in_features=self.num_flow_mods*self.num_feats,
                                                               out_features=1, bias=False)                
                elif self.num_rgbdiff_mods > 0:
                    self.attention_weighting_embed = nn.Linear(in_features=self.num_rgbdiff_mods*self.num_feats,
                                                               out_features=1, bias=False)  
                
    def _prepare_tsn(self):
        
        std = 0.001
        
        if self.num_rgb_mods > 0:    
            if self.embed:
                self.rgb_embed_fc = nn.Linear(self.num_rgb_mods*self.num_feats+self.extra_feats, 300)
                normal(self.rgb_embed_fc.weight, 0, std)
                constant(self.rgb_embed_fc.bias, 0)        
        if self.num_flow_mods > 0:
            if self.embed:
                self.flow_embed_fc = nn.Linear(self.num_flow_mods*self.num_feats, 300)
                normal(self.flow_embed_fc.weight, 0, std)
                constant(self.flow_embed_fc.bias, 0)        
        if self.num_depth_mods > 0:
            if self.embed:
                self.depth_embed_fc = nn.Linear(self.num_depth_mods*self.num_feats, 300)
                normal(self.depth_embed_fc.weight, 0, std)
                constant(self.depth_embed_fc.bias, 0)                  
        if self.num_diff_mods > 0:
            if self.embed:
                self.diff_embed_fc = nn.Linear(self.num_diff_mods*self.num_feats, 300)
                normal(self.diff_embed_fc.weight, 0, std)
                constant(self.diff_embed_fc.bias, 0)          
        
        if self.num_rgb_mods > 0:
            self.cat_fc_rgb = nn.Linear(self.num_rgb_mods*self.num_feats + self.extra_feats, self.num_classes)
            normal(self.cat_fc_rgb.weight, 0, std)
            constant(self.cat_fc_rgb.bias, 0)
        if self.num_flow_mods > 0:
            self.cat_fc_flow = nn.Linear(self.num_flow_mods*self.num_feats, self.num_classes)
            normal(self.cat_fc_flow.weight, 0, std)
            constant(self.cat_fc_flow.bias, 0)    
        if self.num_depth_mods > 0:
            self.cat_fc_depth = nn.Linear(self.num_depth_mods*self.num_feats, self.num_classes)
            normal(self.cat_fc_depth.weight, 0, std)
            constant(self.cat_fc_depth.bias, 0)           
        if self.num_diff_mods > 0:
            self.cat_fc_diff = nn.Linear(self.num_diff_mods*self.num_feats, self.num_classes)
            normal(self.cat_fc_diff.weight, 0, std)
            constant(self.cat_fc_diff.bias, 0)            
        
        if self.num_rgb_mods > 0:
            self.cont_fc_rgb = nn.Linear(self.num_rgb_mods*self.num_feats + self.extra_feats, self.num_dimensions)
            normal(self.cont_fc_rgb.weight, 0, std)
            constant(self.cont_fc_rgb.bias, 0)
        if self.num_flow_mods > 0:
            self.cont_fc_flow = nn.Linear(self.num_flow_mods*self.num_feats, self.num_dimensions)
            normal(self.cont_fc_flow.weight, 0, std)
            constant(self.cont_fc_flow.bias, 0)            
        if self.num_depth_mods > 0:
            self.cont_fc_depth = nn.Linear(self.num_depth_mods*self.num_feats, self.num_dimensions)
            normal(self.cont_fc_depth.weight, 0, std)
            constant(self.cont_fc_depth.bias, 0)   
        if self.num_diff_mods > 0:
            self.cont_fc_diff = nn.Linear(self.num_diff_mods*self.num_feats, self.num_dimensions)
            normal(self.cont_fc_diff.weight, 0, std)
            constant(self.cont_fc_diff.bias, 0)        
    
    
    def _prepare_context_model(self, arch, pretrained):
        
        places_model = torchvision.models.__dict__[arch](num_classes=365)        
        if pretrained:
            model_file = '/gpu-data2/jpik/%s_places365.pth.tar' % arch
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            places_model.load_state_dict(state_dict)
            if self.logger:
                self.logger.info('Initializing RGB context model with Places365 pretrained weights...') 
        modules = list(places_model.children())[:-1] # delete the last fc layer.
        context_model = nn.Sequential(*modules)                
        return context_model    

    
    def _prepare_body_model(self, arch, pretrained):
        
        model = getattr(torchvision.models, arch)(pretrained = pretrained)
        if pretrained:
            if self.logger:
                self.logger.info('Initializing RGB body model with ImageNet pretrained weights...') 
        modules = list(model.children())[:-1] # delete the last fc layer.
        body_model = nn.Sequential(*modules)             
        return body_model
    
    
    def _prepare_face_model(self, arch, pretrained):
        
        if arch != 'resnet18':
            arch = 'resnet50'
            
        resnet_affectnet = Face_Model_18() if arch == 'resnet18' else Face_Model_50()
        if pretrained:    
            pretrained_dict = torch.load('/home/jpik/face_model_resnet18.pt') if arch == 'resnet18' else torch.load('/home/jpik/face_model.pt') 
            resnet_affectnet.load_state_dict(pretrained_dict) # <-- load pretrained weights
            if self.logger:
                self.logger.info('Initializing RGB face model with AffectNet pretrained weights...') 
        modules = list(resnet_affectnet.children())[:-2] # <-- remove FC layers for classification & regression
        face_model = nn.Sequential(*modules)  
        return face_model
    
    
    def train(self, mode=True):
           
        #Override the default train() to freeze the BN parameters
        #:return:
    
        super(TSN, self).train(mode)
        
        if self._enable_pbn:
            
            if self.rgb_body:
                count = 0
                print("Freezing BatchNorm2D modules except the first one in RGB Body Model...")
                for m in self.rgb_body_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
            
            if self.rgb_context:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in RGB Context Model...")
                for m in self.rgb_context_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False            
            
            if self.rgb_face:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in RGB Face Model...")
                for m in self.rgb_face_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False   
                            
            if self.flow_body:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in Flow Body Model...")
                for m in self.flow_body_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False             
            
            if self.flow_context:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in Flow Context Model...")
                for m in self.flow_context_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False             
            
            if self.flow_face:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in Flow Face Model...")
                for m in self.flow_face_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False          
            
            if self.depth:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in Depth Model...")
                for m in self.depth_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False   

            if self.rgbdiff_body:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in RGBDiff Body Model...")
                for m in self.rgbdiff_body_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False                               
            
            if self.rgbdiff_context:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in RGBDiff Context Model...")
                for m in self.rgbdiff_context_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False     
            
            if self.rgbdiff_face:
                count = 0                
                print("Freezing BatchNorm2D modules except the first one in RGBDiff Face Model...")
                for m in self.rgbdiff_face_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False     
    
    
    def get_optim_policies(self):
        params = [{'params': self.parameters()}]
        return params

     
    def forward(self, inputs, num_segments):                 
        
        feats = {}         
        
        if self.rgb_body:
            rgb_body_inp = inputs['body'].view((-1, 3*self.rgb_length) + inputs['body'].size()[-2:])
            x1 = self.rgb_body_model(rgb_body_inp).squeeze(3).squeeze(2)        
            feats['body'] = x1
        
        if self.rgb_context:
            rgb_context_inp = inputs['context'].view((-1, 3*self.rgb_length) + inputs['context'].size()[-2:])
            x2 = self.rgb_context_model(rgb_context_inp).squeeze(3).squeeze(2)
            feats['context'] = x2
        
        if self.rgb_face:
            rgb_face_inp = inputs['face'].view((-1, 3*self.rgb_length) + inputs['face'].size()[-2:])
            x3 = self.rgb_face_model(rgb_face_inp).squeeze(3).squeeze(2)
            feats['face'] = x3      
            
        if self.scenes or self.attributes:
            self.features_blobs[rgb_context_inp.device]=[]              
        
        if self.scenes and self.attributes:
            scenes = self.unified_places_model(rgb_context_inp)
            x4 = F.softmax(scenes, 1)
            fc_feats = self.features_blobs[rgb_context_inp.device][0].double().squeeze(3)
            attributes = torch.matmul(self.W_attribute, fc_feats).float().squeeze(2)
            x5 = F.softmax(attributes, 1) 
            feats['scenes'] = x4
            feats['attributes'] = x5  
        
        if self.scenes and not self.attributes:
            scenes = self.unified_places_model(rgb_context_inp)
            x4 = F.softmax(scenes, 1)  
            feats['scenes'] = x4                  
        
        if not self.scenes and self.attributes:
            scenes = self.unified_places_model(rgb_context_inp)
            fc_feats = self.features_blobs[rgb_context_inp.device][0].double().squeeze(3)
            attributes = torch.matmul(self.W_attribute, fc_feats).float().squeeze(2)
            x5 = F.softmax(attributes, 1)  
            feats['attributes'] = x5                        
        
        if self.flow_body:
            flow_body_inp = inputs['body'].view((-1, 2*self.flow_length) + inputs['body'].size()[-2:])
            x6 = self.flow_body_model(flow_body_inp).squeeze(3).squeeze(2)
            feats['body'] = x6                    
        
        if self.flow_context:
            flow_context_inp = inputs['context'].view((-1, 2*self.flow_length) + inputs['context'].size()[-2:])
            x7 = self.flow_context_model(flow_context_inp).squeeze(3).squeeze(2)
            feats['context'] = x7                                
            
        if self.flow_face:
            flow_face_inp = inputs['face'].view((-1, 2*self.flow_length) + inputs['face'].size()[-2:])
            x8 = self.flow_face_model(flow_face_inp).squeeze(3).squeeze(2)
            feats['face'] = x8                                                     
            
        if self.depth:
            raise NotImplementedError    
        
        if self.rgbdiff_body:
            rgbdiff_body_inp = inputs['body'].view((-1, 3*self.diff_length) + inputs['body'].size()[-2:])
            x10 = self.rgbdiff_body_model(rgbdiff_body_inp).squeeze() 
            feats['body'] = x10                                  
        
        if self.rgbdiff_context:
            rgbdiff_context_inp = inputs['context'].view((-1, 3*self.diff_length) + inputs['context'].size()[-2:])
            x11 = self.rgbdiff_context_model(rgbdiff_context_inp).squeeze()  
            feats['context'] = x11                                                      
        
        if self.rgbdiff_face:
            rgbdiff_face_inp = inputs['face'].view((-1, 3*self.diff_length) + inputs['face'].size()[-2:])
            x12 = self.rgbdiff_face_model(rgbdiff_face_inp).squeeze()              
            feats['face'] = x12                                                                      
        
        if self.num_rgb_mods > 0:
            if self.num_rgb_mods > 1:
                feats_fused = torch.cat(list(feats.values()), dim = 1)
            else:
                feats_fused = list(feats.values())[0]    
        elif self.num_flow_mods > 0:
            if self.num_flow_mods > 1:
                feats_fused = torch.cat(list(feats.values()), dim = 1)
            else:
                feats_fused = list(feats.values())[0]    
        elif self.num_diff_mods > 0:
            if self.num_diff_mods > 1:
                feats_fused = torch.cat(list(feats.values()), dim = 1)
            else:
                feats_fused = list(feats.values())[0]    
        else:
            raise NotImplementedError    
            
        if self.consensus_type=='attention_weighting':
            attention_weights_cat = self.attention_weighting_cat(feats_fused.view((-1, num_segments) + feats_fused.size()[1:])).squeeze(2)
            attention_weights_cat = F.softmax(attention_weights_cat, 1).unsqueeze(2)
            attention_weights_cont = self.attention_weighting_cont(feats_fused.view((-1, num_segments) + feats_fused.size()[1:])).squeeze(2)
            attention_weights_cont = F.softmax(attention_weights_cont, 1).unsqueeze(2)
            if self.embed:
                attention_weights_embed = self.attention_weighting_embed(feats_fused.view((-1, num_segments) + feats_fused.size()[1:])).squeeze(2)
                attention_weights_embed = F.softmax(attention_weights_embed, 1).unsqueeze(2)        
        
        outputs = {}

        if self.embed:
            if self.num_rgb_mods > 0:
                rgb_embed_segm = self.rgb_embed_fc(feats_fused)
                rgb_embed = rgb_embed_segm.view((-1, num_segments) + rgb_embed_segm.size()[1:])
                if self.consensus_type=='avg':                
                    rgb_embed = self.consensus_embed(rgb_embed).squeeze(1)
                elif self.consensus_type=='linear_weighting':
                    rgb_embed = self.linear_weighting_embed(torch.transpose(rgb_embed, 1, 2)).squeeze(2)
                elif self.consensus_type=='attention_weighting':
                    rgb_embed = torch.matmul(torch.transpose(rgb_embed, 1, 2), attention_weights_embed).squeeze(2)  
                outputs['embeddings'] = rgb_embed
                
            if self.num_flow_mods > 0:
                flow_embed_segm = self.flow_embed_fc(feats_fused)
                flow_embed = flow_embed_segm.view((-1, num_segments) + flow_embed_segm.size()[1:])
                if self.consensus_type=='avg':
                    flow_embed = self.consensus_embed(flow_embed).squeeze(1)
                elif self.consensus_type=='linear_weighting':
                    flow_embed = self.linear_weighting_embed(torch.transpose(flow_embed, 1, 2)).squeeze(2)
                elif self.consensus_type=='attention_weighting':  
                    flow_embed = torch.matmul(torch.transpose(flow_embed, 1, 2), attention_weights_embed).squeeze(2)                      
                outputs['embeddings'] = flow_embed
            
            if self.num_depth_mods > 0:
                depth_embed_segm = self.depth_embed_fc(x9)
                depth_embed = depth_embed_segm.view((-1, num_segments) + depth_embed_segm.size()[1:])
                depth_embed = self.consensus_embed(depth_embed).squeeze(1)
                outputs['embeddings'] = depth_embed

            if self.num_diff_mods > 0:
                diff_embed_segm = self.diff_embed_fc(feats_fused)
                diff_embed = diff_embed_segm.view((-1, num_segments) + diff_embed_segm.size()[1:])
                diff_embed = self.consensus_embed(diff_embed).squeeze(1)
                outputs['embeddings'] = diff_embed                
                
        if self.num_rgb_mods > 0:
            out_cat = self.cat_fc_rgb(feats_fused)
        if self.num_flow_mods > 0:
            out_cat = self.cat_fc_flow(feats_fused)
        if self.num_depth_mods > 0:
            out_cat = self.cat_fc_depth(x9)        
        if self.num_diff_mods > 0:
            out_cat = self.cat_fc_diff(feats_fused)         
        
        if self.num_rgb_mods > 0:
            out_cont = self.cont_fc_rgb(feats_fused)
        if self.num_flow_mods > 0:
            out_cont = self.cont_fc_flow(feats_fused)
        if self.num_depth_mods > 0:
            out_cont = self.cont_fc_depth(x9)        
        if self.num_diff_mods > 0:
            out_cont = self.cont_fc_diff(feats_fused) 
            
        out_cat = out_cat.view((-1, num_segments) + out_cat.size()[1:])
        out_cont = out_cont.view((-1, num_segments) + out_cont.size()[1:])
        
        if self.consensus_type=='avg':
            output_cat = self.consensus_cat(out_cat).squeeze(1)
            output_cont = self.consensus_cont(out_cont).squeeze(1)
        elif self.consensus_type=='linear_weighting':
            output_cat = self.linear_weighting_cat(torch.transpose(out_cat, 1, 2)).squeeze(2)
            output_cont = self.linear_weighting_cont(torch.transpose(out_cont, 1, 2)).squeeze(2)
        elif self.consensus_type=='attention_weighting':
            output_cat = torch.matmul(torch.transpose(out_cat, 1, 2), attention_weights_cat).squeeze(2)
            output_cont = torch.matmul(torch.transpose(out_cont, 1, 2), attention_weights_cont).squeeze(2)            
            
        outputs['categorical'] = output_cat        
        outputs['continuous'] = output_cont

        return outputs


    def _construct_flow_model(self, arch, mode, pretrained):
        
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        
        if mode == 'body':
            model = getattr(torchvision.models, arch)(pretrained = pretrained)
            if pretrained:
                if self.logger:
                    self.logger.info('Initializing Flow Body model with ImageNet pretrained weights...')
            modules = list(model.children())[:-1] # delete the last fc layer.
            base_model = nn.Sequential(*modules)         
        elif mode == 'context':
            model = getattr(torchvision.models, arch)(pretrained = pretrained)
            if pretrained:
                if self.logger:
                    self.logger.info('Initializing Flow Context model with ImageNet pretrained weights...')
            modules = list(model.children())[:-1] # delete the last fc layer.
            base_model = nn.Sequential(*modules)     
        else:
            model = getattr(torchvision.models, arch)(pretrained = pretrained)
            if pretrained:
                if self.logger:
                    self.logger.info('Initializing Flow Face model with ImageNet pretrained weights...')
            modules = list(model.children())[:-1] # delete the last fc layer.
            base_model = nn.Sequential(*modules)     
        
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.flow_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.flow_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        
        return base_model

    
    def _construct_depth_model(self, arch, pretrained):
        
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        
        model = getattr(torchvision.models, arch)(pretrained = pretrained)
        if pretrained:
            if self.logger:
                self.logger.info('Initializing Depth model with ImageNet pretrained weights...')
        modules = list(model.children())[:-1] # delete the last fc layer.
        base_model = nn.Sequential(*modules)         
        
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (self.depth_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(self.depth_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        
        return base_model    
    
    
    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data
    
    
    def _construct_diff_model(self, arch, mode, pretrained, keep_rgb=False):
        
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        
        if mode == 'body':
            model = getattr(torchvision.models, arch)(pretrained = pretrained)
            if pretrained:
                if self.logger:
                    self.logger.info('Initializing RGBDiff body model with ImageNet pretrained weights...')
            modules = list(model.children())[:-1] # delete the last fc layer.
            base_model = nn.Sequential(*modules)         
        elif mode == 'context':
            model = getattr(torchvision.models, arch)(pretrained = pretrained)
            if pretrained:
                if self.logger:
                    self.logger.info('Initializing RGBDiff context model with ImageNet pretrained weights...')
            modules = list(model.children())[:-1] # delete the last fc layer.
            base_model = nn.Sequential(*modules)     
        else:
            model = getattr(torchvision.models, arch)(pretrained = pretrained)
            if pretrained:
                if self.logger:
                    self.logger.info('Initializing RGBDiff face model with ImageNet pretrained weights...')
            modules = list(model.children())[:-1] # delete the last fc layer.
            base_model = nn.Sequential(*modules)  
                
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.diff_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.diff_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.diff_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model
 

    def hook_feature(self, module, input, output):    
        self.features_blobs[input[0].device].append(output) 
        
        
    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.rgb_body_model.parameters(), 'lr': lr*lrp},
            {'params': self.rgb_context_model.parameters(), 'lr': lr*lrp},
            {'params': self.rgb_face_model.parameters(), 'lr': lr*lrp},
            {'params': self.cont_fc.parameters(), 'lr': lr},
            {'params': self.cat_fc.parameters(), 'lr': lr}
        ]        