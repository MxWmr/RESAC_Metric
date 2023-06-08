import xarray as xr 
import os
import torch
from tqdm import tqdm
import sys
from scipy.ndimage import uniform_filter1d
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
save_path = "/usr/home/mwemaere/neuro/Data3/"





def split_sets(ssh_array,year=2017):
    n_leap = (year-1993)//4
    ssh_test = ssh_array[365*(year-1993)+n_leap:365*(year+1-1993)+n_leap,:,:].clone()
    ssh_train = torch.concat((ssh_array[:365*(year-1993)+n_leap,:,:],ssh_array [(year+1-1993)+n_leap:,:,:]),axis=0)

    return ssh_train,ssh_test



def pool_images(ssh_aray,pool):
    return torch.squeeze(pool(torch.unsqueeze(ssh_array,1)))



def save_test(ssh_test,mean,std,inp=False,sst=False,u=False,v=False,sat=False):

    ssh_test -= mean
    ssh_test /= std
    ssh_test = torch.unsqueeze(ssh_test.to(torch.float32),1)
    if inp:
        torch.save(ssh_test,save_path+"test_ssh_in"+".pt")
    elif sst:
        torch.save(ssh_test,save_path+"test_sst"+".pt")    
    elif u:
        torch.save(ssh_test,save_path+"test_u"+".pt")    
    elif v:
        torch.save(ssh_test,save_path+"test_v"+".pt")        
    elif sat:
        torch.save(ssh_test,save_path+"test_ssh_sat"+".pt")  
    else:
        torch.save(ssh_test,save_path+"test_ssh_out"+".pt")


def save_valid(ssh_valid,mean,std,inp=False,sst=False,u=False,v=False):

    ssh_valid = torch.unsqueeze(ssh_valid.to(torch.float32),1)
    if inp:
        torch.save(ssh_valid,save_path+"valid_ssh_in"+".pt")
    if sst:
        torch.save(ssh_valid,save_path+"valid_sst"+".pt")
    if u:
        torch.save(ssh_valid,save_path+"valid_u"+".pt")
    if v:
        torch.save(ssh_valid,save_path+"valid_v"+".pt")
    else:
        torch.save(ssh_valid,save_path+"valid_ssh_out"+".pt")



def save_train(ssh_array,mean,std,rand_perm,inp=False,sst=False,u=False,v=False):

    # shuffle data 
    ssh_array = ssh_array[rand_perm]

    n=0
    for i in tqdm(range(0,len(ssh_array),100)):
        ssh = ssh_array[i:i+100,:,:].clone()
        ssh = torch.unsqueeze(ssh,1)
        ssh = ssh.to(torch.float32)

        if inp:
            torch.save(ssh,save_path+"ssh_in_"+str(n)+".pt")
        elif sst:
            torch.save(ssh,save_path+"sst_"+str(n)+".pt")
        elif u:
            torch.save(ssh,save_path+"u_"+str(n)+".pt")
        elif v:
            torch.save(ssh,save_path+"v_"+str(n)+".pt")
        else:
            torch.save(ssh,save_path+"ssh_out_"+str(n)+".pt")
        n+=1





if True:    



    ##### formating output data 
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['sla'].values).to(torch.float32)

    ssh_in = ssh_array.clone()

    ssh_train,ssh_test = split_sets(ssh_in,year=2019)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std
    torch.save(torch.tensor([mean,std]),save_path+"mean_std_out"+".pt")



    ##save test set
    save_test(ssh_test,mean,std)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2018)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std)

    # define sequence of id for shuffling
    rand_perm = torch.randperm(ssh_train.size()[0])

    # normalize and save train set
    save_train(ssh_train,mean,std,rand_perm)    # use same rand_prem than sat !

    


  


    ##### formating input data 

    #define pool layer to size the mod data
    pool = torch.nn.AvgPool2d(3,stride=(3,3))

    ## pool images
    ssh_array = pool_images(ssh_array,pool)


    ## split train and test set
    ssh_train,ssh_test = split_sets(ssh_array,year=2019)


    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std


    ##save test set
    save_test(ssh_test,mean,std,inp=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2018)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,inp=True)

    # normalize and save train set
    save_train(ssh_train,mean,std,rand_perm,inp=True)    # use same rand_prem than sat !

    



    ##### formating sst data 
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['sst'].values).to(torch.float32)

    ssh_in = ssh_array.clone()

    ssh_train,ssh_test = split_sets(ssh_in,year=2019)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std


    ##save test set
    save_test(ssh_test,mean,std,sst=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2018)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,sst=True)


    # normalize and save train set
    save_train(ssh_train,mean,std,rand_perm,sst=True)    # use same rand_prem than sat !

    


     ##### formating u data 
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['uo'].values).to(torch.float32)

    ssh_in = ssh_array.clone()

    ssh_train,ssh_test = split_sets(ssh_in,year=2019)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std


    ##save test set
    save_test(ssh_test,mean,std,u=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2018)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,u=True)


    # normalize and save train set
    save_train(ssh_train,mean,std,rand_perm,u=True)    # use same rand_prem than sat !

    



    ##### formating v data 
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/GLORYS12V1_PRODUCT_001_030/"

    files = "glorys12v1_mod_product_001_030_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['vo'].values).to(torch.float32)

    ssh_in = ssh_array.clone()

    ssh_train,ssh_test = split_sets(ssh_in,year=2019)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std


    ##save test set
    save_test(ssh_test,mean,std,v=True)

    ## split train and valid set
    ssh_train,ssh_valid= split_sets(ssh_train,year=2018)   

    # normalize and save valid set
    save_valid(ssh_valid,mean,std,v=True)


    # normalize and save train set
    save_train(ssh_train,mean,std,rand_perm,v=True)    # use same rand_prem than sat !

    



    ##### formating sat test data 
    data_path = "/data/labo/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/SSH_PRODUCT_008_047/"

    files = "ssh_sat_product_008_047_*.nc"


    filenames = os.path.join(data_path,files)

    mf_ds = xr.open_mfdataset(filenames)


    ssh_array = torch.tensor(mf_ds['sla'].values).to(torch.float32)

    ssh_in = ssh_array.clone()

    ssh_train,ssh_test = split_sets(ssh_in,year=2019)

    ## get and save mean and std
    mean = torch.mean(ssh_train)
    ssh_train-=mean
    std = torch.std(ssh_train) 
    ssh_train/=std


    ##save test set
    save_test(ssh_test,mean,std,sat=True)

