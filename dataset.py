import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF


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


class TSNDataset(data.Dataset):
    
    def __init__(self, mode,
                 num_segments=3, 
                 inp_type='RGB',
                 rgb_transform=None,
                 flow_transform=None,
                 depth_transform=None,
                 diff_transform=None,
                 random_shift=True, 
                 context=True):

        # Change the template accordingly
        self.rgb_tmpl = "img_{:05d}.jpg"
        self.flow_tmpl = "{}_{:05d}.jpg"
        
        self.num_segments = num_segments
        self.rgb_transform = rgb_transform
        self.flow_transform = flow_transform
        self.depth_transform = depth_transform
        self.diff_transform = diff_transform
        self.random_shift = random_shift
        self.test_mode = (mode=='test')
        self.inp_type = inp_type

        # Change the path accordingly
        self.bold_path = "/gpu-data2/jpik/BoLD/BOLD_public"

        self.context = context
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

        # Change the path accordingly
        self.embeddings = np.load("glove_840B_embeddings.npy")
        
        if inp_type=='RGB':
            self.data_length = 5
        elif inp_type=='Flow':
            self.data_length = 5
        elif inp_type=='RGBDiff':
            self.data_length = 6
        elif inp_type=='Depth':
            raise NotImplementedError  
    
        
    def get_context(self, image, joints, format="cv2"):
        
        joints = joints.reshape((18,3))
        joints[joints[:,2]<0.1] = np.nan
        joints[np.isnan(joints[:,2])] = np.nan

        joint_min_x = int(round(np.nanmin(joints[:,0])))
        joint_min_y = int(round(np.nanmin(joints[:,1])))

        joint_max_x = int(round(np.nanmax(joints[:,0])))
        joint_max_y = int(round(np.nanmax(joints[:,1])))

        expand_x = int(round(10/100 * (joint_max_x-joint_min_x)))
        expand_y = int(round(10/100 * (joint_max_y-joint_min_y)))

        if format == "cv2":
            
            image[max(0, joint_min_x - expand_x):min(joint_max_x + expand_x, image.shape[1])] = [0,0,0]
        
        elif format == "PIL":
            
            bottom = min(joint_max_y+expand_y, image.height)
            right = min(joint_max_x+expand_x,image.width)
            top = max(0,joint_min_y-expand_y)
            left = max(0,joint_min_x-expand_x)
            image = np.array(image)
            
            if len(image.shape) == 3:
                image[top:bottom,left:right] = [0,0,0]
            else:
                image[top:bottom,left:right] = np.min(image)
            return Image.fromarray(image)


    def get_bounding_box(self, image, joints, format="cv2"):
        
        joints = joints.reshape((18,3))
        joints[joints[:,2]<0.1] = np.nan
        joints[np.isnan(joints[:,2])] = np.nan

        joint_min_x = int(round(np.nanmin(joints[:,0])))
        joint_min_y = int(round(np.nanmin(joints[:,1])))

        joint_max_x = int(round(np.nanmax(joints[:,0])))
        joint_max_y = int(round(np.nanmax(joints[:,1])))

        expand_x = int(round(100/100 * (joint_max_x-joint_min_x)))
        expand_y = int(round(100/100 * (joint_max_y-joint_min_y)))

        if format == "cv2":
            return image[max(0,joint_min_y-expand_y):min(joint_max_y+expand_y, image.shape[0]), max(0,joint_min_x-expand_x):min(joint_max_x+expand_x,image.shape[1])]
        elif format == "PIL":
            bottom = min(joint_max_y+expand_y, image.height)
            right = min(joint_max_x+expand_x,image.width)
            top = max(0,joint_min_y-expand_y)
            left = max(0,joint_min_x-expand_x)
            return tF.crop(image, top, left, bottom-top ,right-left)
        
    
    def get_face(self, image, joints, _modality, format="cv2"):
        
        joints = joints.reshape((18,3))
        #joints[joints[:,2]<0.1] = np.nan
        #joints[np.isnan(joints[:,2])] = np.nan
        
        head_indices = [0,1,14,15,16,17]
        
        joints = joints[head_indices]
    
        if not np.isnan(joints).all():

            joint_min_x = int(round(np.nanmin(joints[:,0])))
            joint_min_y = int(round(np.nanmin(joints[:,1])))

            joint_max_x = int(round(np.nanmax(joints[:,0])))
            joint_max_y = int(round(np.nanmax(joints[:,1])))

            expand_x = int(round(30/100 * (joint_max_x-joint_min_x)))
            expand_y = int(round(30/100 * (joint_max_y-joint_min_y)))

            if format == "cv2":
                return image[max(0,joint_min_y-expand_y):min(joint_max_y+expand_y, image.shape[0]), max(0,joint_min_x-expand_x):min(joint_max_x+expand_x,image.shape[1])]
            elif format == "PIL":
                bottom = min(joint_max_y+expand_y, image.height)
                right = min(joint_max_x+expand_x,image.width)
                top = max(0,joint_min_y-expand_y)
                left = max(0,joint_min_x-expand_x)
                return tF.crop(image, top, left, bottom-top ,right-left)
        else:
            if _modality == 'RGB':
                return Image.new('RGB', (300, 300))
            else:
                return Image.new('L', (300, 300))
        

    def joints(self, index):
        
        sample = self.df.iloc[index]
        joints_path = os.path.join(self.bold_path, "joints", sample["joints_path"])
        joints18 = np.load(joints_path)
        joints18[:,0] -= joints18[0,0]
        return joints18

    
    def _load_image(self, directory, idx, index, modality, mode):
        
        joints = self.joints(index)
        poi_joints = joints[joints[:, 0] + 1 == idx]
        sample = self.df.iloc[index]
        poi_joints = poi_joints[(poi_joints[:, 1] == sample["person_id"]), 2:]

        if modality == 'RGB':

            frame = Image.open(os.path.join(directory, self.rgb_tmpl.format(idx))).convert("RGB")
          
            if mode == "context":
                if poi_joints.size == 0:
                    return [frame]
                context = self.get_context(frame, poi_joints, format="PIL")
                return [context]
            
            if poi_joints.size == 0:
                body = frame
                face = Image.new('RGB', (300, 300))
                pass #do whole frame, black face image 
            else:
                body = self.get_bounding_box(frame, poi_joints, format="PIL") 
                face = self.get_face(frame, poi_joints, _modality='RGB', format="PIL")
                if body.size == 0:
                    print(poi_joints)
                    body = frame
                    face = Image.new('RGB', (300, 300))               
            return [body], [face]
        
        elif modality == 'Flow':
            
            frame_x = Image.open(os.path.join(directory, self.flow_tmpl.format('flow_x', idx))).convert('L')
            frame_y = Image.open(os.path.join(directory, self.flow_tmpl.format('flow_y', idx))).convert('L')
    
            if mode == "context":
                if poi_joints.size == 0:
                    return [frame_x, frame_y]
                context_x = self.get_context(frame_x, poi_joints, format="PIL")
                context_y = self.get_context(frame_y, poi_joints, format="PIL")
                return [context_x, context_y]

            if poi_joints.size == 0:
                body_x = frame_x
                body_y = frame_y
                face_x = Image.new('L', (300, 300))
                face_y = Image.new('L', (300, 300))
                pass # do whole frame
            else:
                body_x = self.get_bounding_box(frame_x, poi_joints, format="PIL")
                body_y = self.get_bounding_box(frame_y, poi_joints, format="PIL")
                face_x = self.get_face(frame_x, poi_joints, _modality="Flow", format="PIL")
                face_y = self.get_face(frame_y, poi_joints, _modality="Flow", format="PIL")        
                if body_x.size == 0:
                    body_x = frame_x
                    body_y = frame_y
                    face_x = Image.new('L', (300, 300))
                    face_y = Image.new('L', (300, 300))                    
            return [body_x, body_y], [face_x, face_y]
        
        elif modality == 'Depth':
            
            # Change it accordingly
            directory_depth = directory.replace('test_raw','depth')
            
            frame = Image.open(os.path.join(directory, self.rgb_tmpl.format(idx))).convert("RGB")
            
            depth_map = Image.open(os.path.join(directory_depth, self.rgb_tmpl.format(idx))).convert("L")
            
            return [frame], [depth_map]
        
        elif modality == 'RGBDiff':
            
            frame = Image.open(os.path.join(directory, self.rgb_tmpl.format(idx))).convert("RGB")
            
            return [frame]
            
    
    def _load_diffimage(self, frame, idx, index, mode):
        
        joints = self.joints(index)
        poi_joints = joints[joints[:, 0] + 1 == idx]
        sample = self.df.iloc[index]
        poi_joints = poi_joints[(poi_joints[:, 1] == sample["person_id"]), 2:]
                  
        if mode == "context":
            if poi_joints.size == 0:
                return [frame]
            context = self.get_context(frame, poi_joints, format="PIL")
            return [context]
            
        if poi_joints.size == 0:
            body = frame
            face = Image.new('RGB', (300, 300))
            pass # do whole frame, black face image 
        else:
            body = self.get_bounding_box(frame, poi_joints, format="PIL") 
            face = self.get_face(frame, poi_joints, _modality='RGB', format="PIL")
            if body.size == 0:
                print(poi_joints)
                body = frame
                face = Image.new('RGB', (300, 300))               
        return [body], [face]


    def _sample_indices(self, record):
        
        """
        :param record: VideoRecord
        :return: list
        """
        
        average_duration = (record.num_frames - self.data_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration-1, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.data_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 3

    
    def _get_val_indices(self, record):
                
        if record.num_frames > self.num_segments + self.data_length - 1:
            tick = (record.num_frames - self.data_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 3

    
    def _get_test_indices(self, record):
        
        tick = (record.num_frames - self.data_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 3

    
    def __getitem__(self, index):
                
        sample = self.df.iloc[index]

        fname = os.path.join(self.bold_path,"videos",self.df.iloc[index]["video"])

        capture = cv2.VideoCapture(fname)
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
        
        capture.release()

        record_path = os.path.join(self.bold_path,"test_raw",sample["video"][4:-4])
        
        record = VideoRecord([record_path, frame_count, sample["min_frame"], sample["max_frame"]])

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
            
        return self.get(record, segment_indices, index)

    
    def get(self, record, indices, index):
                
        rgb_body = list() 
        rgb_context = list() 
        rgb_face = list()
        
        flow_body = list()
        flow_context = list()
        flow_face = list()
        
        frame = list()
        #depth = list()
        
        rgbdiff = list()
        rgbdiff_body = list()
        rgbdiff_context = list()
        rgbdiff_face = list()
        
        #print(indices)
        
        """ Get RGB """
        if self.inp_type=="RGB":
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(1):
                    seg_body, seg_face = self._load_image(record.path, p, index, modality = "RGB", mode = "body")
                    rgb_body.extend(seg_body)
                    rgb_face.extend(seg_face)
                    if self.context:
                        seg_context = self._load_image(record.path, p, index, modality = "RGB", mode = "context")
                        rgb_context.extend(seg_context)
                    if p < record.num_frames:
                        p += 1

        """ Get Optical Flow """ 
        if self.inp_type=="Flow":
            for seg_ind in indices:  
                p = int(seg_ind)-2
                for i in range(5):
                    seg_body, seg_face = self._load_image(record.path, p, index, modality = "Flow", mode = "body")
                    flow_body.extend(seg_body)
                    flow_face.extend(seg_face)
                    if self.context:
                        seg_context = self._load_image(record.path, p, index, modality = "Flow", mode = "context")
                        flow_context.extend(seg_context)
                    if p < record.num_frames:
                        p += 1     
                    
        #""" Get Depth """ 
        #if self.inp_type=="Depth":
        #    for seg_ind in indices:  
        #       p = int(seg_ind)-2
        #       for i in range(self.data_length):
        #           seg_frame, seg_depth = self._load_image(record.path, p, index, modality = "Depth", mode = None)
        #           frame.extend(seg_frame)
        #           depth.extend(seg_depth)
        #           if p < record.num_frames:
        #               p += 1              
        
        """ Get RGBDiff"""
        if self.inp_type=="RGBDiff":
            for seg_ind in indices:
                p = int(seg_ind)-2
                for i in range(6):
                    seg_frame = self._load_image(record.path, p, index, modality = "RGBDiff", mode = None)
                    frame.extend(seg_frame)
                    if p < record.num_frames:
                        p += 1        
        
            for j in range(1,self.data_length):
                diff = (np.array(frame[j])-np.array(frame[j-1]))
                diff = (diff.astype('float64')-np.min(diff))/(np.max(diff)-np.min(diff))
                diff = 255*diff
                diff = diff.astype('uint8')
                rgbdiff.append(diff)
            
            for seg_ind in indices:
                p = int(seg_ind)-2
                for i in range(5): 
                    diff_body, diff_face = self._load_diffimage(Image.fromarray(rgbdiff[i]), p, index, mode = 'body') 
                    rgbdiff_body.extend(diff_body)
                    rgbdiff_face.extend(diff_face)
                    if self.context:
                        diff_context = self._load_diffimage(Image.fromarray(rgbdiff[i]), p, index, mode = 'context')
                        rgbdiff_context.extend(diff_context)
                    if p < record.num_frames:
                        p += 1 
    
        if not self.test_mode: 
            categorical = self.df.iloc[index][self.categorical_emotions]
            continuous = self.df.iloc[index][self.continuous_emotions]
            continuous = continuous/10.0 # normalize to 0 - 1

            if self.inp_type=="RGB":
                if self.rgb_transform is None:
                    process_rgb_body = rgb_body
                    if self.context:
                        process_rgb_context = rgb_context
                    process_rgb_face = rgb_face
                else:
                    process_rgb_body = self.rgb_transform(rgb_body)
                    if self.context:
                        process_rgb_context = self.rgb_transform(rgb_context)
                    process_rgb_face = self.rgb_transform(rgb_face)
            if self.inp_type=="Flow":
                if self.flow_transform is None:
                    process_flow_body = flow_body
                    if self.context:
                        process_flow_context = flow_context
                    process_flow_face = flow_face
                else:
                    process_flow_body = self.flow_transform(flow_body)
                    if self.context:
                        process_flow_context = self.flow_transform(flow_context)
                    process_flow_face = self.flow_transform(flow_face)
            if self.inp_type=="RGBDiff":
                if self.diff_transform is None:
                    process_rgbdiff_body = rgbdiff_body
                    if self.context:
                        process_rgbdiff_context = rgbdiff_context                                
                    process_rgbdiff_face = rgbdiff_face
                else:
                    process_rgbdiff_body = self.diff_transform(rgbdiff_body)
                    if self.context:
                        process_rgbdiff_context = self.diff_transform(rgbdiff_context)                
                    process_rgbdiff_face = self.diff_transform(rgbdiff_face)
            
            if self.inp_type=="Flow":
                if self.context:
                    return process_flow_body, process_flow_face, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"], process_flow_context            
                else:
                    return process_flow_body, process_flow_face, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"]            
            if self.inp_type=="RGB":
                if self.context:
                    return process_rgb_body, process_rgb_face, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"], process_rgb_context             
                else:
                    return process_rgb_body, process_rgb_face, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"]
            if self.inp_type=="RGBDiff":
                if self.context:
                    return process_rgbdiff_body, process_rgbdiff_face, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"], process_rgbdiff_context  
                else:
                    return process_rgbdiff_body, process_rgbdiff_face, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"]  

        # --> Testing (TODO)
        else:   
            if self.inp_type=="RGB":
                if self.rgb_transform is None:
                    process_rgb_body = rgb_body
                    if self.context:
                        process_rgb_context = rgb_context
                    process_rgb_face = rgb_face
                else:
                    process_rgb_body = self.rgb_transform(rgb_body)
                    if self.context:
                        process_rgb_context = self.rgb_transform(rgb_context)
                    process_rgb_face = self.rgb_transform(rgb_face)
            if self.inp_type=="Flow":
                if self.flow_transform is None:
                    process_flow_body = flow_body
                    if self.context:
                        process_flow_context = flow_context
                    process_flow_face = flow_face
                else:
                    process_flow_body = self.flow_transform(flow_body)
                    if self.context:
                        process_flow_context = self.flow_transform(flow_context)
                    process_flow_face = self.flow_transform(flow_face)
            if self.inp_type=="RGBDiff":
                if self.diff_transform is None:
                    process_rgbdiff_body = rgbdiff_body
                    if self.context:
                        process_rgbdiff_context = rgbdiff_context                                
                    process_rgbdiff_face = rgbdiff_face
                else:
                    process_rgbdiff_body = self.diff_transform(rgbdiff_body)
                    if self.context:
                        process_rgbdiff_context = self.diff_transform(rgbdiff_context)                
                    process_rgbdiff_face = self.diff_transform(rgbdiff_face)

            if self.inp_type=="Flow":
                if self.context:
                    return process_flow_body, process_flow_face, torch.tensor(self.embeddings).float(), self.df.iloc[index]["video"], process_flow_context            
                else:
                    return process_flow_body, process_flow_face, torch.tensor(self.embeddings).float(), self.df.iloc[index]["video"]            
            if self.inp_type=="RGB":
                if self.context:
                    return process_rgb_body, process_rgb_face, torch.tensor(self.embeddings).float(), self.df.iloc[index]["video"], process_rgb_context      
                else:
                    return process_rgb_body, process_rgb_face, torch.tensor(self.embeddings).float(), self.df.iloc[index]["video"]      
            if self.inp_type=="RGBDiff":
                if self.context:
                    return process_rgbdiff_body, process_rgbdiff_face, torch.tensor(self.embeddings).float(), self.df.iloc[index]["video"], process_rgbdiff_context  
                else:
                    return process_rgbdiff_body, process_rgbdiff_face, torch.tensor(self.embeddings).float(), self.df.iloc[index]["video"]  
    

    def __len__(self):
        return len(self.df)
