import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from dataset import VideosDataset,VideosDataset2
from torch.utils.data import DataLoader
import numpy as np
#import wandb
#wandb.login()


class Net(nn.Module):

    def __init__(self,num_of_classes,num_of_features_e,num_of_features_k, simple_mode=True,weight1 = 0.5):
        super(Net, self).__init__()
        self.simple_mode= simple_mode
        self.fc1 = nn.Linear(num_of_features_k,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,num_of_classes)
        
        self.fc4 = nn.Linear(num_of_features_e,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,num_of_classes)
        
        
        self.dr = nn.Dropout(0.5)
        self.weight = weight1

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x=F.softmax(x)
        return x
    
    def forward(self,x,y):
        x = torch.flatten(x, -1)
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.dr(self.fc2(x)))
        x = F.softmax(self.fc3(self.dr(x)))
        
        y = torch.flatten(y, -1)
        y = F.relu(self.fc4(y.float()))
        y = F.relu(self.dr(self.fc5(y)))
        y = F.softmax(self.fc6(self.dr(y)))
        
        z=x*self.weight+y*(1-self.weight)
        return z


def test_pred(test_loader,model,criterion):
    gesture_mapping={"G0":0,"G1":1,"G2":2,"G3":3,"G4":4,"G5":5}
    labels_dict=list(gesture_mapping.keys())
    preds=[]
    y=[]
    test_loss=0
    accr=[]
    for i, data in enumerate(test_loader):
        input_e,input_k, labels = data

        labels = labels.long()
        outputs = model(input_k,input_e)
        pred=torch.argmax(outputs).item()

        preds.append(pred)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        accr.append(pred==labels.item())
    #wandb.log({"test loss": test_loss/len(test_loader)})
    acc=sum(accr)/len(accr)
    #wandb.log({"accracy": acc})
    return test_loss/len(test_loader), acc


def train_no_vid(all_train_data,all_nets,weight,device,criterion,filepath='/home/student/cvop_project/models/'):
    
    all_optimizers = []
    for net in all_nets:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        all_optimizers.append(optimizer)
    for epoch in range(10):
        train_loss = 0
        running_loss = 0.0
        for j in range(len(all_train_data)):
            print("epoch: ", epoch, " fold: ", j)
            model = all_nets[j]
            optimizer = all_optimizers[j]
            train_data = all_train_data[j][0]
            valid_data = all_train_data[j][1]
            train_loader = DataLoader(train_data, batch_size=64 ,shuffle=True)
                
            valid_loader = DataLoader(valid_data, batch_size=64 ,shuffle=False)
            
            for i, data in enumerate(train_loader):  # fix shuffle
                try:
                    input_e,input_k, labels = data
                    input_e,input_k, labels=input_e.to(device),input_k.to(device), labels.to(device)

                    labels = labels.long()
                    optimizer.zero_grad()
                    outputs = model(input_k,input_e)
                    
                    loss = criterion(outputs, labels)


                    loss.backward()
                    optimizer.step()
                    if i % 2000 == 1999:
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                        running_loss = 0.0
                    running_loss += loss.item()
                    train_loss += loss.item()
                except Exception as e:
                    print(e)
            #wandb.log({"train loss": (train_loss/len(train_loader))})
            valid_loss = 0
            for i, data in enumerate(valid_loader):
                input_e,input_k, labels = data
                input_e,input_k, labels=input_e.to(device),input_k.to(device), labels.to(device)

                labels = labels.long()
                optimizer.zero_grad()
                outputs = model(input_k,input_e)
                    
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
            print("validation loss: " ,valid_loss/len(valid_loader))
            #wandb.log({"valid loss": valid_loss/len(valid_loader)})
    print('Finished Training')

    for i,net in enumerate(all_nets):
        torch.save(net.state_dict(),filepath+"{:.1f}".format(weight)+"/"+str(i)+".pt")

    
    





def main():
    DATA_DIR= "../../../datashare/"

    train_vid=False
    test= True
   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    all_fold_files = os.listdir(DATA_DIR+"/APAS/folds")
    all_test_files = [f for f in all_fold_files if "test" in f]
    all_valid_files = [f for f in all_fold_files if "valid" in f]

    print(os.getcwd())





    criterion = nn.CrossEntropyLoss()


    if test:
        #wandb.init()
        weight_value=0.1

        model_path="models/0.1/0.pt"
        foldNum="0"

        test_data = VideosDataset2(str(0),all_valid_files[0],all_test_files[0],"test",preload=True,videos_cap=-1)
        e_feautres,k_features,num_of_classes=test_data.input_param
        test_loader=DataLoader(test_data, batch_size=1,shuffle=False,num_workers=8)
        net = Net(num_of_classes=num_of_classes,num_of_features_e=e_feautres,num_of_features_k=k_features,weight1=weight_value)
        net.load_state_dict(torch.load(model_path))
        model=net.eval()
        avg_loss,acc=test_pred(test_loader,model,criterion)
        print(f"average loss: {avg_loss} Accracy: {acc}")
        

        
        #wandb.finish()
        
    else:
        if train_vid:
            train_data = VideosDataset("0",all_valid_files[0],all_test_files[0],"train")
            e_feautres,k_features,num_of_classes=train_data.input_param

            #net = GreyNet(num_of_classes=num_of_classes,num_of_features_e=e_feautres,num_of_features_k=k_features).to(device)
            #train_with_vid(train_data,net,filepath='/home/student/cvop_project/models/model_test2.pt')


        else:
            for weight_value in [0.1]:#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                #wandb.init()
                print("dataset:")
                all_train_data = []
                for i in range(1):
                    train_data = VideosDataset2(str(i),all_valid_files[i],all_test_files[i],"train",preload=True,videos_cap=-1)
                    valid_data = VideosDataset2(str(i),all_valid_files[i],all_test_files[i],"valid",preload=True,videos_cap=-1)
                    all_train_data.append([train_data,valid_data])
                e_feautres,k_features,num_of_classes=train_data.input_param
                
                all_nets = []
                for i in range(1):
                    net = Net(num_of_classes=num_of_classes,num_of_features_e=e_feautres,num_of_features_k=k_features,weight1=weight_value).to(device)
                    all_nets.append(net)
                    #wandb.watch(net, log_freq=100)
                #print(net)
                #train_dataloader_videos = DataLoader(train_data, batch_size=64 ,shuffle=True) 
                print("train_loader")
                train_no_vid(all_train_data,all_nets,weight_value,device,criterion)
                #wandb.finish()

if __name__ == "__main__":
    main()