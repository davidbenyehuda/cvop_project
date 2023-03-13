import os
#import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
#from PIL import Image
from torchvision import transforms,datasets

DATA_DIR= "../../../datashare/"

class VideosDataset(Dataset):
    def __init__(self, fold_num, valid_file, test_file, stage, transform=None, target_transform=None,videos_cap=-1):
        self.fold_num=fold_num
        print(valid_file,test_file)
        f = open(DATA_DIR+"APAS/folds/"+valid_file, "r")
        valid_files = f.readlines()
        valid_files = [f.split('.')[0] for f in valid_files]
        f.close()
        f = open(DATA_DIR+"APAS/folds/"+test_file, "r")
        test_files = f.readlines()
        test_files = [f.split('.')[0] for f in test_files]
        f.close()
        all_files = os.listdir(DATA_DIR+"APAS/features/fold"+fold_num)
        files_for_stage = []
        for f in all_files:
            name = f.split(".")[0]
            if stage == "train" and name not in valid_files and name not in test_files:
                files_for_stage.append(f)
            if stage == "valid" and name in valid_files:
                files_for_stage.append(f)
            if stage == "test" and name in test_files:
                files_for_stage.append(f)
        files_for_stage= [x for x  in files_for_stage if x not in ['P032_tissue2.npy','P025_balloon2.npy']]       
        files_for_stage =files_for_stage[:videos_cap] #TODO: need to be removed 
        
        labels = []
        all_features_npy = []
        all_kinematics_npy = []
        gesture_mapping = {"G0":0,"G1":1,"G2":2,"G3":3,"G4":4,"G5":5}
        self.gesture_mapping= gesture_mapping 
        self.input_param= [1280, 36,6]
        self.files_for_stage=files_for_stage
        # for f in files_for_stage:
        #      if f not in ['P032_tissue2.npy','P025_balloon2.npy']:
        #         print(f)
        #         data = np.load(DATA_DIR+"APAS/features/fold"+fold_num+"/"+f)
        #         data = np.transpose(data)
            
        #         n_f=data.shape[0]
            
        #         data_k = np.load(DATA_DIR+"APAS/kinematics_npy/"+f)
        #         data_k = np.transpose(data_k)
        #         k_f=data_k.shape[0]
        #         missing_frames_number = n_f-k_f
        #         if missing_frames_number > 0:
        #             missing_frames=np.random.choice(range(k_f),missing_frames_number)
        #             data_k=np.insert(data_k,missing_frames,[data_k[i,:] for i in missing_frames],axis=0) 
        #         elif missing_frames_number < 0:
        #             missing_frames=np.random.choice(range(n_f),-missing_frames_number)
        #             data=np.insert(data,missing_frames,[data[i,:] for i in missing_frames],axis=0)
            
            
        #         #frames_vec=torch.cat((torch.from_numpy(data),torch.from_numpy(data_k)),axis=0)
        #         all_features_npy.extend(torch.from_numpy(data))
        #         all_kinematics_npy.extend(torch.from_numpy(data_k))
                 
        #         '''frames_path=DATA_DIR+"APAS/frames/"+f.split('.')[0]+"_top"                  
        #         frames= sorted(os.listdir(frames_path))
        #         framesdataset=CustomImageDataset(img_names=frames,img_dir=frames_path)
        #         all_videos.extend(framesdataset)'''
                
                
                
                
        #         with open(DATA_DIR+"APAS/transcriptions_gestures/"+f.split('.')[0]+".txt", "r") as file:
        #             labels_data = file.readlines()
        #         for line in labels_data:
        #             values = line[:-1].split()
        #             if int(values[1]) > len(data):
        #                 ul = len(data)
        #             else:
        #                 ul = values[1]
        #             if int(values[0]) == 0:
        #                 start = 1
        #             else:
        #                 start = int(values[0])
                
        #             for i in range(start,int(ul)+1):
        #                 labels.append(gesture_mapping[values[2]])
        #             #print(start,int(ul),len(labels))
        #         #break
        #         #print(len(all_features_npy),len(labels))
        # self.features_embbeding  =torch.stack(all_features_npy)
        # self.features_kinematics =torch.stack(all_kinematics_npy)
        # self.labels = labels
        # #self.input_param=[self.features_embbeding.shape[-1],self.features_kinematics.shape[-1],len(gesture_mapping)]
        
        # print(len(self.features_embbeding),len(self.features_kinematics),len(self.labels))
    
    
    
    def __len__(self):
        return len(self.files_for_stage)

    def __getitem__(self, idx):
        
        return VideoFrameDataset(self.fold_num,self.files_for_stage[idx],self.gesture_mapping)
        
        #return self.features_embbeding[idx],self.features_kinematics[idx],self.labels[idx]
    
class CustomImageDataset(Dataset):
    def __init__(self,img_names ,img_dir, transform= transforms.ToTensor(), target_transform=None):
        self.img_labels =img_names
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image=Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image
    
class VideoFrameDataset(Dataset):
    def __init__(self,fold_num,f,gesture_mapping) -> None:
        super().__init__()
        self.fold_num=fold_num
        self.file_name=f
        self.gesture_mapping= gesture_mapping

        if f not in ['P032_tissue2.npy','P025_balloon2.npy']:
            data = np.load(DATA_DIR+"APAS/features/fold"+self.fold_num+"/"+f)
            data = np.transpose(data)
            
            n_f=data.shape[0]
            
            data_k = np.load(DATA_DIR+"APAS/kinematics_npy/"+f)
            data_k = np.transpose(data_k)
            k_f=data_k.shape[0]
            missing_frames_number = n_f-k_f
            if missing_frames_number > 0:
                missing_frames=np.random.choice(range(k_f),missing_frames_number)
                data_k=np.insert(data_k,missing_frames,[data_k[i,:] for i in missing_frames],axis=0) 
            elif missing_frames_number < 0:
                missing_frames=np.random.choice(range(n_f),-missing_frames_number)
                data=np.insert(data,missing_frames,[data[i,:] for i in missing_frames],axis=0)
             #frames_vec=torch.cat((torch.from_numpy(data),torch.from_numpy(data_k)),axis=0)
            data=torch.from_numpy(data)
            data_k=torch.from_numpy(data_k)
            labels=[]
            frames_path=DATA_DIR+"APAS/frames/"+f.split('.')[0]+"_top/"                  
            frames= sorted(os.listdir(frames_path))
            
            
            
            #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                         std=[0.229, 0.224, 0.225])

            framesdataset=CustomImageDataset(img_names=frames,img_dir=frames_path,
                                             transform=transforms.Compose([
            #        transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
            #        normalize,
                ]))
           
            
            with open(DATA_DIR+"APAS/transcriptions_gestures/"+f.split('.')[0]+".txt", "r") as file:
                labels_data = file.readlines()
            for line in labels_data:
                values = line[:-1].split()
                if int(values[1]) > len(data):
                    ul = len(data)
                else:
                    ul = values[1]
                if int(values[0]) == 0:
                    start = 1
                else:
                    start = int(values[0])
                
                for i in range(start,int(ul)+1):
                    labels.append(self.gesture_mapping[values[2]])    
           
            missing_frames_number = len(labels)-len(framesdataset)
            if  missing_frames_number > 0:
                    missing_frames=np.random.choice(len(framesdataset),missing_frames_number)
                    for index in missing_frames:
                        framesdataset.img_labels.insert(index,framesdataset.img_labels[index])
                    
                    
                    
                    
                
                
           
            
            self.vid=framesdataset
            self.f_e=data
            self.f_k=data_k 
            self.labels=labels
        print(f,len(framesdataset),len(labels))
        assert len(framesdataset) ==len(labels)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
       
        #return self.f_k[idx,:],self.f_e[idx,:],self.labels[idx]

        return self.vid[idx],self.f_k[idx,:],self.f_e[idx,:],self.labels[idx]
            
class VideosDataset2(Dataset):
    def __init__(self, fold_num, valid_file, test_file, stage, transform=None, target_transform=None,preload=False,videos_cap=-1):
        self.fold_num=fold_num
        self.preload=preload
        gesture_mapping = {"G0":0,"G1":1,"G2":2,"G3":3,"G4":4,"G5":5}
        self.gesture_mapping= gesture_mapping 
        self.input_param= [1280, 36,6]
        print(valid_file,test_file)
        f = open(DATA_DIR+"APAS/folds/"+valid_file, "r")
        valid_files = f.readlines()
        valid_files = [f.split('.')[0] for f in valid_files]
        f.close()
        f = open(DATA_DIR+"APAS/folds/"+test_file, "r")
        test_files = f.readlines()
        test_files = [f.split('.')[0] for f in test_files]
        f.close()
        all_files = os.listdir(DATA_DIR+"APAS/features/fold"+fold_num)
        files_for_stage = []
        for f in all_files:
            name = f.split(".")[0]
            if stage == "train" and name not in valid_files and name not in test_files:
                files_for_stage.append(f)
            if stage == "valid" and name in valid_files:
                files_for_stage.append(f)
            if stage == "test" and name in test_files:
                files_for_stage.append(f)
                
        #files_for_stage =files_for_stage[:5] #TODO: need to be removed 
        files_for_stage=files_for_stage[:videos_cap]
        files_for_stage= [x for x  in files_for_stage if x not in ['P032_tissue2.npy','P025_balloon2.npy']]
        self.files_for_stage=files_for_stage       
        
        if preload:
            all_features_npy = []
            all_kinematics_npy = []
            labels = []
            
            for f in files_for_stage:
                if f not in ['P032_tissue2.npy','P025_balloon2.npy']:
                    data,data_k,label=self.get_features(f)
                    all_features_npy.extend(data)
                    all_kinematics_npy.extend(data_k)
                    labels.extend(label)
            self.features_embbeding  =torch.stack(all_features_npy)
            self.features_kinematics =torch.stack(all_kinematics_npy)
            self.labels = labels
        
            
            print(len(self.features_embbeding),len(self.features_kinematics),len(self.labels),)


    def get_features(self,f):
        
        if f not in ['P032_tissue2.npy','P025_balloon2.npy']:
            data = np.load(DATA_DIR+"APAS/features/fold"+self.fold_num+"/"+f)
            data = np.transpose(data)
            
            n_f=data.shape[0]
            
            data_k = np.load(DATA_DIR+"APAS/kinematics_npy/"+f)
            data_k = np.transpose(data_k)
            k_f=data_k.shape[0]
            missing_frames_number = n_f-k_f
            if missing_frames_number > 0:
                missing_frames=np.random.choice(range(k_f),missing_frames_number)
                data_k=np.insert(data_k,missing_frames,[data_k[i,:] for i in missing_frames],axis=0) 
            elif missing_frames_number < 0:
                missing_frames=np.random.choice(range(n_f),-missing_frames_number)
                data=np.insert(data,missing_frames,[data[i,:] for i in missing_frames],axis=0)
             #frames_vec=torch.cat((torch.from_numpy(data),torch.from_numpy(data_k)),axis=0)
            data=torch.from_numpy(data)
            data_k=torch.from_numpy(data_k)
            labels=[]
           
            with open(DATA_DIR+"APAS/transcriptions_gestures/"+f.split('.')[0]+".txt", "r") as file:
                labels_data = file.readlines()
            for line in labels_data:
                values = line[:-1].split()
                if int(values[1]) > len(data):
                    ul = len(data)
                else:
                    ul = values[1]
                if int(values[0]) == 0:
                    start = 1
                else:
                    start = int(values[0])
                
                for i in range(start,int(ul)+1):
                    labels.append(self.gesture_mapping[values[2]])    

            return data,data_k,labels
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        if self.preload:
            feature_e = self.features_embbeding[idx]
            feature_k = self.features_kinematics[idx]
            label = self.labels[idx]
        else:
             feature_e,feature_k,label=self.get_features(self.files_for_stage[idx])
        return feature_e,feature_k,label
