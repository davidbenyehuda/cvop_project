# %%
import numpy as np
import os
import cv2
import os
#import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image
from torchvision import transforms,datasets
from model import Net
import matplotlib.pyplot as plt
import wandb

DATA_DIR= "../../../datashare/"


# %%
class CustomImageDataset(Dataset):
    def __init__(self,img_names ,img_dir, transform= transforms.ToTensor(), target_transform=None):
        self.img_labels =img_names
        self.img_dir = img_dir
        self.transform = transform
        #self.transform=None
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image= cv2.imread(img_path)
        if self.transform:
            img = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image


class VideoFrameDataset(Dataset):
    def __init__(self,fold_num,f,top=True) -> None:
        super().__init__()
        self.fold_num=fold_num
        self.file_name=f
        self.gesture_mapping= {"G0":0,"G1":1,"G2":2,"G3":3,"G4":4,"G5":5}
        self.topOrSide="_"+("top" if top else "side")
        self.shape=[1280, 36,6]

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
            frames_path=DATA_DIR+"APAS/frames/"+f.split('.')[0]+f"{self.topOrSide}/"                  
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

# %%
def load_model(model,path,args):
    model = model(*args)
    model.load_state_dict(torch.load(path))
    return model.eval()

def predict_with_video(model,dataset,videoname,fps=30,folder="videos"):
    gesture_mapping={"G0":0,"G1":1,"G2":2,"G3":3,"G4":4,"G5":5}
    labels_dict=list(gesture_mapping.keys())
    x,y,w,h = 0,0,400,80
    img_array=[]
    i=0
    for data in dataset:
        frame,input_e,input_v,label=data
        frame=frame.squeeze()

        height, width, layers = frame.shape
        size = (width, height)
        pred=model(input_e,input_v)
        pred=torch.argmax(pred).item()
        
        # Window name in which image is displayed        
        # text
        text = "Pred:" +labels_dict[pred]+"   "+ "Label:" +labels_dict[label]
        
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        # org
        org = (x + int(w / 15), y + int(h / 3))
        # fontScalew2
        fontScale =1
        # Red color in BGR
        color = (0, 0, 255) 
        # Line thickness of 2 px
        thickness = 1
        
        image = cv2.putText(frame.numpy(), text, org, font, fontScale, 
                 color, thickness, cv2.LINE_AA)
        # plt.imshow(image)
        # plt.savefig(videoname.split(".")[0]+'.png')

        # plt.show()
        #cv2.imshow("window",image)

        # cv2.putText(img=frame, text="Pred:" +labels_dict[pred]+ "Label:" +labels_dict[label], org=(x + int(w / 15), y + int(h / 3)),
        #         fontFace=cv2.FONT_HERSHEY_DUPLEX,
        #         fontScale=0.5, color=(0, 0, 255), thickness=1)
        img_array.append(image)
        i+=1
        if i%500==0:
            print(f"at frame:{i}")
    out = cv2.VideoWriter(f'{folder}/{videoname.split(".")[0]}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release() 



def create_vid(
    foldNum="0"
    ,top=True
    ,videoname="P016_tissue1.npy"
    ,model_path="models/0.1/0.pt"
    ,weight_value=0.1):


        vid=VideoFrameDataset(f=videoname,fold_num=foldNum,top=top)
        vid_loader=DataLoader(vid, batch_size=1,shuffle=False,num_workers=8)
        num_of_features_e,num_of_features_k,num_of_classes=vid.shape

        net = Net(num_of_classes=num_of_classes,num_of_features_e=num_of_features_e,num_of_features_k=num_of_features_k,weight1=weight_value)
        net.load_state_dict(torch.load(model_path))
        model=net.eval()
        predict_with_video(model,vid_loader,videoname=videoname)


# %%
def main():
    all_fold_files = os.listdir(DATA_DIR+"/APAS/folds")
    all_files=os.listdir(DATA_DIR+"APAS/features/fold0")[:10]
    i=0
    for file in all_files:
        side= i%2==0 
        create_vid(videoname=file,top=side)
        i+=1
    
if __name__ == "__main__":
    main()
# %%

def predict(model,data_loader,frames_loader,labels_dict):
    x,y,w,h = 0,0,400,25
    img_array=[]
    for data,frame in zip(data_loader,frames_loader):
        input_e,input_v,label=data
        height, width, layers = frame.shape
        size = (width, height)
        pred=model(input_e,input_v)
        labels_dict[pred]
        cv2.putText(img=frame, text="Pred:" +labels_dict[pred]+ "Label:" +labels_dict[label], org=(x + int(w / 15), y + int(h / 3)),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5, color=(0, 0, 255), thickness=1)
        img_array.append(frame)
    out = cv2.VideoWriter('P026_tissue1_new.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
            

