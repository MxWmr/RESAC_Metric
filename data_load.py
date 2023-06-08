import torch
import math as mt
import numpy as np
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self,l_files,n_files,path,file_name_in,file_name_sst,file_name_out,file_name_u,file_name_v,batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.file_name_in = file_name_in
        self.file_name_sst = file_name_sst
        self.file_name_u = file_name_u
        self.file_name_v = file_name_v
        self.file_name_out = file_name_out
        self.l_files = l_files
        self.path = path
        self.n_files = n_files

    def __len__(self):
        return (self.l_files//self.batch_size + 1)*self.n_files

    def __getitem__(self,i):
        leng  = (self.l_files//self.batch_size + 1)*self.n_files
        
        i_f = i//(self.l_files//self.batch_size + 1)
        i_2 = i % (self.l_files//self.batch_size +1)

        if i >= leng:
            raise IndexError()

        d_in = torch.load(self.path+self.file_name_in+str(i_f)+'.pt')[:,:,:,:88]
        d_sst = torch.load(self.path+self.file_name_sst+str(i_f)+'.pt')[:,:,:,:264]
        d_u = torch.load(self.path+self.file_name_u+str(i_f)+'.pt')[:,:,:,:264]
        d_v = torch.load(self.path+self.file_name_v+str(i_f)+'.pt')[:,:,:,:264]
        d_out = torch.load(self.path+self.file_name_out+str(i_f)+'.pt')[:,:,:,:264]

        if self.batch_size*(i_2+1) <= self.l_files:
            return tuple([d_in[i_2*self.batch_size:self.batch_size*(i_2+1)],d_sst[i_2*self.batch_size:self.batch_size*(i_2+1)],d_out[i_2*self.batch_size:self.batch_size*(i_2+1)],d_u[i_2*self.batch_size:self.batch_size*(i_2+1)],d_v[i_2*self.batch_size:self.batch_size*(i_2+1)]])
        else:
            return tuple([d_in[i_2*self.batch_size:],d_sst[i_2*self.batch_size:],d_out[i_2*self.batch_size:],d_u[i_2*self.batch_size:],d_v[i_2*self.batch_size:],])


class ConcatData(torch.utils.data.Dataset):
    def __init__(self,datasets,shuffle=False,batch_size=1):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size

        if shuffle:
            n = len(datasets[0])
            id_rd = torch.randperm(n)
            for d in self.datasets:
                d = d[list(id_rd)]

    def __getitem__(self,i):
        self.datasets[0][(i+1)*self.batch_size]
        return tuple(d[i*self.batch_size:(i+1)*self.batch_size] for d in self.datasets)


    def __len__(self):
        return min(int(len(d)/self.batch_size) for d in self.datasets)


