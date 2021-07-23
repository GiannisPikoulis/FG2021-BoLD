import torch.utils.data as data
import cv2
import os
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF
from tools.tools import *


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


class VideoRecord(object):
    
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    
    @property
    def min_frame(self):
        return int(self._data[2])

    @property
    def max_frame(self):
        return int(self._data[3])


class SkeletonDataset(data.Dataset):
    
    def __init__(self, mode, 
                 normalize=True,
                 centralize=False,
                 random_choose=False,
                 random_move=False,
                 random_shift=False):

        self.bold_path = "/gpu-data2/jpik/BoLD/BOLD_public"
        self.test_mode = (mode=='test')
        self.mode = mode

        self.categorical_emotions = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence", "Happiness",
                                     "Pleasure", "Excitement", "Surprise", "Sympathy", "Doubt/Confusion", "Disconnect",
                                     "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance", "Anger",
                                     "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain", "Suffering"]

        self.continuous_emotions = ["Valence", "Arousal", "Dominance"]

        self.attributes = ["Gender", "Age", "Ethnicity"]

        header = ["video", "person_id", "min_frame", "max_frame"] + self.categorical_emotions + self.continuous_emotions + self.attributes + ["annotation_confidence"]
        
        if not self.test_mode:
            self.df = pd.read_csv(os.path.join(self.bold_path, "annotations/{}.csv".format(mode)), names=header)
        else:
            self.df = pd.read_csv(os.path.join(self.bold_path, "annotations/test_meta.csv"), names=header)        
        
        self.df["joints_path"] = self.df["video"].apply(rreplace,args=[".mp4",".npy",1])
        self.video_list = self.df["video"]
        
        self.random_choose = random_choose
        self.random_move = random_move
        self.random_shift = random_shift
        self.normalize = normalize
        self.centralize = centralize
        self.T = 297 # max joint sequence length 
        
        """
        Max joint coordinates per dimension,
        per frame, as found within each fold of the 
        BoLD dataset.
        Change paths accordingly. 
        """
        if mode == "train":
            self.max_x = np.load("/home/jpik/NTUA-BEEU-eccv2020-master/BOLD_train_max_x_joint.npy")
            self.max_y = np.load("/home/jpik/NTUA-BEEU-eccv2020-master/BOLD_train_max_y_joint.npy")
        elif mode == "val":
            self.max_x = np.load("/home/jpik/NTUA-BEEU-eccv2020-master/BOLD_val_max_x_joint.npy")
            self.max_y = np.load("/home/jpik/NTUA-BEEU-eccv2020-master/BOLD_val_max_y_joint.npy") 
        elif mode == 'test':
            self.max_x = np.load("/home/jpik/NTUA-BEEU-eccv2020-master/BOLD_test_max_x_joint.npy")
            self.max_y = np.load("/home/jpik/NTUA-BEEU-eccv2020-master/BOLD_test_max_y_joint.npy")             
        
           
    def joints(self, index):
        
        sample = self.df.iloc[index]
        joints_path = os.path.join(self.bold_path, "joints", sample["joints_path"])
        joints18 = np.load(joints_path)
        joints18[:,0] -= joints18[0,0]
        return joints18

    
    def _load_joints(self, directory, idx, index):
        
        joints = self.joints(index)

        poi_joints = joints[joints[:, 0] + 1 == idx]
        sample = self.df.iloc[index]
        poi_joints = poi_joints[(poi_joints[:, 1] == sample["person_id"]), 2:]
        
        if poi_joints.size == 0:
            poi_joints = np.zeros((18,3))
        else:    
            poi_joints = poi_joints.reshape((18,3))
        
        poi_joints[poi_joints[:,2]<0.1] = np.nan
        poi_joints[np.isnan(poi_joints[:,2])] = np.nan
        
        return poi_joints
    
    
    def __getitem__(self, index):
              
        sample = self.df.iloc[index]

        fname = os.path.join(self.bold_path,"videos",self.df.iloc[index]["video"])
        
        capture = cv2.VideoCapture(fname)
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
        
        capture.release()

        record_path = os.path.join(self.bold_path,"test_raw",sample["video"][4:-4])

        record = VideoRecord([record_path, frame_count, sample["min_frame"], sample["max_frame"]])
   
        return self.get(record, index)

    
    def get(self, record, index):
        
        joints = list()
    
        for ind in range(1, record.num_frames):
            p = int(ind)           
            j = self._load_joints(record.path, p, index)
            j[np.isnan(j)] = 0
            
            if self.normalize:
                j[:,0] = j[:,0]/float(self.max_x[index])
                j[:,1] = j[:,1]/float(self.max_y[index])
            
            if self.centralize:
                j[:,0] = j[:,0]-0.5
                j[:,1] = j[:,1]-0.5
                
            joints.append(np.transpose(j))
                
        if not self.test_mode: 
            categorical = self.df.iloc[index][self.categorical_emotions]
            continuous = self.df.iloc[index][self.continuous_emotions]
            continuous = continuous/10.0 # normalize to 0 - 1
        
        joints = np.stack(joints, axis=1)
        joints = np.array(np.expand_dims(joints, axis=-1))
        
        if self.random_shift:
            joints = random_shift(joints)
        if self.random_choose:
            joints = random_choose(joints, self.T)
        else:
            joints = auto_padding(joints, self.T, random_pad=(self.mode=="train"))
        if self.random_move:
            joints = random_move(joints)
        
        if self.mode != 'test':
            return torch.tensor(joints).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float()
        else:
            return torch.tensor(joints).float()
        
        
    def __len__(self):
        return len(self.df)