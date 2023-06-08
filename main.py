
import torch
from datetime import datetime
import numpy as np 
from archi import *
from plot_utils import *
from data_load import *



if torch.cuda.is_available():
    device = "cuda:1" 
else:
    raise('No GPU !')

date = datetime.now().strftime("%m_%d_%H:%M_")

# load data
save_path = '/usr/home/mwemaere/neuro/resac_metric/Save/'


data_path = '/usr/home/mwemaere/neuro/Data3/'



train_loader = Dataset(100,270,data_path,'ssh_in_','sst_','ssh_out_','u_','v_',batch_size=32)


test_in = torch.load(data_path + 'test_ssh_in.pt')[:,:,:,:88]
test_sst = torch.load(data_path + 'test_sst.pt')[:,:,:,:264]
test_out = torch.load(data_path + 'test_ssh_out.pt')[:,:,:,:264]
test_u= torch.load(data_path + 'test_u.pt')[:,:,:,:264]
test_v= torch.load(data_path + 'test_v.pt')[:,:,:,:264]

test_loader = ConcatData([test_in,test_sst,test_out,test_u,test_v],shuffle=False)



criterion = RMSELoss()

model = resac2()

if False:    #train
    lr = 1e-3  
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    num_epochs = 80

    model.fit(device,optimizer,criterion,train_loader,num_epochs)

    torch.save(model.state_dict(), save_path+date+'model.pth')

if True:   #test
    device= 'cuda:0'
    date = '06_07_19:10_'
    model.load_state_dict(torch.load(save_path+date+'model.pth'))
    model = model.to(device)

    mean,std, l_im = model.test(criterion,test_loader,device, get_im=[291,28,102,464])


    print(mean)
    print(std)
    with open('test_result.txt', 'a') as f:
        f.write('\n'+date+'\n')
        f.write(str(mean)+'\n')
        f.write(str(std)+'\n')

        f.close()

    plot_test_ssh(l_im,save_path,date)
