import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class resac(nn.Module):
    def __init__(self):
        super().__init__()
        
        nf = 36

        l_conv1 = []
        l_conv1.append(nn.Conv2d(2,nf,3,padding='same'))

        for i in range(8):
            l_conv1.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv1.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv1 = nn.ModuleList(l_conv1)



        nf = 24



        l_conv2 = []
        l_conv2.append(nn.Conv2d(2,nf,3,padding='same'))

        for i in range(6):
            l_conv2.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv2.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv2 = nn.ModuleList(l_conv2)




        l_conv3 = []
        l_conv3.append(nn.Conv2d(1,nf,3,padding='same'))

        for i in range(6):
            l_conv3.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv3.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv3 = nn.ModuleList(l_conv3)



        l_conv4 = []
        l_conv4.append(nn.Conv2d(1,nf,3,padding='same'))

        for i in range(6):
            l_conv4.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv4.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv4 = nn.ModuleList(l_conv4)


        self.upsamp = nn.Upsample(scale_factor=3,mode='bicubic')
        self.bn1 = nn.BatchNorm2d(36)
        self.bn =  nn.BatchNorm2d(nf) 
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()



    def CNN1(self,im):
        for i in range(0,len(self.l_conv1)-2,2):
            im = self.l_conv1[i](im)
            im = self.relu(im)
            im = self.l_conv1[i+1](im)
            im = self.relu(im)
            im = self.bn1(im)

        im = self.l_conv1[-2](im)
        im = self.relu(im)
        im = self.l_conv1[-1](im)
        ssh12 = im

        return ssh12


    def CNN2(self,im):

        for i in range(0,len(self.l_conv2)-2,2):
            im = self.l_conv2[i](im)
            im = self.relu(im)
            im = self.l_conv2[i+1](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l_conv2[-2](im)
        im = self.relu(im)
        im = self.l_conv2[-1](im)
        uv = im

        return uv

    def CNN3(self,im):

        for i in range(0,len(self.l_conv3)-2,2):
            im = self.l_conv3[i](im)
            im = self.relu(im)
            im = self.l_conv3[i+1](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l_conv3[-2](im)
        im = self.relu(im)
        im = self.l_conv3[-1](im)
        u = im
        
        return u


    def CNN4(self,im):

        for i in range(0,len(self.l_conv4)-2,2):
            im = self.l_conv4[i](im)
            im = self.relu(im)
            im = self.l_conv4[i+1](im)
            im = self.relu(im)
            im = self.bn(im)

        im = self.l_conv4[-2](im)
        im = self.relu(im)
        im = self.l_conv4[-1](im)
        v = im

        return v  


    def forward(self,X):
        ssh4,sst12 = X[0],X[1]

        ssh4_up = self.upsamp(ssh4)
        ssh_sst12 = torch.concat((ssh4_up,sst12),axis=1)
        ssh12 = self.CNN1(ssh_sst12)


        ssh_sst_12_bis = torch.concat((ssh12,sst12),axis=1)
        uv_12 = self.CNN2(ssh_sst_12_bis)

        u = self.CNN3(uv_12)
        v = self.CNN4(uv_12)

        return [ssh12,u,v]


    def fit(self,device,optimizer,criterion,train_loader,num_epochs):

        model = self.to(device)
        tbw = SummaryWriter()

        for epoch in range(num_epochs):
            print('epoch: {}'.format(epoch+1))

            l_loss1 = []
            l_loss2 = []
            l_loss3 = []
            
            for (ssh4,sst12,ssh12,u,v) in tqdm(train_loader):



                optimizer.zero_grad()

                X = [ssh4.to(device),sst12.to(device)]
                y = [ssh12.to(device),u.to(device),v.to(device)]

                b_size = X[0].shape[0]

                if b_size == 0:
                    print('aï')
                    continue

                y_pred = model(X)

                loss1 = criterion(y_pred[0],y[0])
                loss2 = criterion(y_pred[1],y[1])
                loss3 = criterion(y_pred[2],y[2])

                loss = loss1 + loss2 + loss3 
                l_loss1.append(loss1.item())
                l_loss2.append(loss2.item())
                l_loss3.append(loss3.item())

                loss.backward()

                optimizer.step()

            tbw.add_scalar("loss 1",np.array(l_loss1).mean(),epoch)
            tbw.add_scalar("loss 2",np.array(l_loss2).mean(),epoch)
            tbw.add_scalar("loss 3",np.array(l_loss3).mean(),epoch)

            if epoch%4 == 0:
                tbw.add_image("prediction ssh12",y_pred[1][0])
                tbw.add_image("target ssh",y[1][0])
                tbw.add_image("prediction u",y_pred[2][0])
                tbw.add_image("target u",y[2][0])

        tbw.close()
            

    def test(self,criterion,test_loader,device,get_im):
        model = self.to(device)
        test_accuracy = []    
        l_im = []
        [mean_mod,std_mod] = torch.load('/usr/home/mwemaere/neuro/Data3/mean_std_out.pt')
        with torch.no_grad():
            for i,(ssh4,sst12,ssh12,u,v) in enumerate(test_loader):
                X = [ssh4.to(device),sst12.to(device)]
                y = [ssh12.to(device),u.to(device),v.to(device)]
                
                b_size = X[0].shape[0]
                if b_size == 0:
                    continue

                y_pred = model(X)
                test_accuracy.append(criterion(y[0]*std_mod+mean_mod,y_pred[0]*std_mod+mean_mod).item())

                if i in get_im:
                    l_im.append([X[0],y_pred[0],y[0]])


        test_accuracy = np.array(test_accuracy)
        mean = np.mean(test_accuracy, axis=0)
        std = np.std(test_accuracy, axis=0)
        if len(get_im)!=0:
            return mean,std,l_im
        else:
            return mean,std
        








class resac2(nn.Module):
    def __init__(self):
        super().__init__()
        
        nf = 36

        l_conv1 = []
        l_conv1.append(nn.Conv2d(2,nf,3,padding='same'))

        for i in range(16):
            l_conv1.append(nn.Conv2d(nf,nf,3,padding='same'))

        l_conv1.append(nn.Conv2d(nf,1,1,padding='same'))
        self.l_conv1 = nn.ModuleList(l_conv1)


        self.upsamp = nn.Upsample(scale_factor=3,mode='bicubic')
        self.bn1 = nn.BatchNorm2d(36)

        self.relu = nn.SiLU()
        self.sig = nn.Sigmoid()



    def CNN1(self,im):
        for i in range(0,len(self.l_conv1)-2,2):
            im = self.l_conv1[i](im)
            im = self.relu(im)
            im = self.l_conv1[i+1](im)
            im = self.relu(im)
            im = self.bn1(im)

        im = self.l_conv1[-2](im)
        im = self.relu(im)
        im = self.l_conv1[-1](im)
        ssh12 = im

        return ssh12



    def forward(self,X):
        ssh4,sst12 = X[0],X[1]

        ssh4_up = self.upsamp(ssh4)
        ssh_sst12 = torch.concat((ssh4_up,sst12),axis=1)
        ssh12 = self.CNN1(ssh_sst12)

        return [ssh12,ssh4_up]


    def fit(self,device,optimizer,criterion,train_loader,num_epochs):

        model = self.to(device)
        tbw = SummaryWriter()
        

        for epoch in range(num_epochs):
            print('epoch: {}'.format(epoch+1))

            l_loss1 = []
            l_loss2 = []

            for (ssh4,sst12,ssh12,u,v) in tqdm(train_loader):



                optimizer.zero_grad()

                X = [ssh4.to(device),sst12.to(device)]
                y = [ssh12.to(device),u.to(device),v.to(device)]

                b_size = X[0].shape[0]

                if b_size == 0:
                    print('aï')
                    continue

                y_pred = model(X)

                loss = criterion(y_pred[0],y[0])
                l_loss1.append(loss.item())

                loss2 = criterion(y_pred[1],y[0])
                l_loss2.append(loss2.item())

                loss.backward()

                optimizer.step()

            tbw.add_scalar("loss 1",np.array(l_loss1).mean(),epoch)
            tbw.add_scalar("loss with b2b",np.array(l_loss2).mean(),epoch)


            if epoch%4 == 0:
                tbw.add_image("prediction ssh12",y_pred[0][0])
                tbw.add_image("target ssh",y[0][0])
                tbw.add_image("b2b",y_pred[1][0])


        tbw.close()
            

    def test(self,criterion,test_loader,device,get_im=[]):
        model = self.to(device)
        test_accuracy = []    
        l_im = []
        [mean_mod,std_mod] = torch.load('/usr/home/mwemaere/neuro/Data3/mean_std_out.pt')
        with torch.no_grad():
            for i,(ssh4,sst12,ssh12,u,v) in tqdm(enumerate(test_loader)):
                X = [ssh4.to(device),sst12.to(device)]
                y = [ssh12.to(device),u.to(device),v.to(device)]
                
                b_size = X[0].shape[0]
                if b_size == 0:
                    continue

                y_pred = model(X)
                test_accuracy.append(criterion(y[0]*std_mod+mean_mod,y_pred[0]*std_mod+mean_mod).item())

                if i in get_im:
                    l_im.append([X[0],y_pred[0],y[0]])


        test_accuracy = np.array(test_accuracy)
        mean = np.mean(test_accuracy, axis=0)
        std = np.std(test_accuracy, axis=0)
        if len(get_im)!=0:
            return mean,std,l_im
        else:
            return mean,std
        





class RMSELoss(torch.nn.Module):
    def __init__(self,coeff=1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.coeff = coeff
        
    def forward(self,yhat,y):
        return self.coeff*torch.sqrt(self.mse(yhat,y))





 






        