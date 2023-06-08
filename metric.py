
import torch
from datetime import datetime
import numpy as np 
from archi import *
from plot_utils import *
from data_load import *




class RESACMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device= 'cpu'
        date = '06_07_19:10_'   
        save_path = '/usr/home/mwemaere/neuro/resac_metric/Save/'   #path of where is the model.pth of RESAC
        model = resac2() 
        model.load_state_dict(torch.load(save_path+date+'model.pth'))
        self.model = model.to(self.device)

    def forward(self,test_loader_sat,test_loader_transformed):
        """
        test_loader_sat: data loader of the sattelite data 365x1x72x88
        test_loader_transformed: data loader of the transformed sat data 365x1x72x88
        # --> should be normalized in mean 0 std 1
        # mean and std should be in a file in the same dir as data named: mean_std_out.pt  as torch.tensor([mean,std]) in order to denormalize 
        # this path can be changed in archi.py in resac2.test
        """

        mean1,std1 = self.model.test(criterion,test_loader_sat,self.device)
        mean2,std2 = self.model.test(criterion,test_loader_transformed,self.device)

        
        # print('resac RMSE with original sat data: {}'.format(mean1))
        # print('resac RMSE with transformed sat data: {}'.format(mean2))
        # print('RESAC Metric:{}'.format(mean1-mean2))


        return mean1-mean2
    




if __name__ == '__main__':

    # load data
    save_path = '/usr/home/mwemaere/neuro/resac_metric/Save/'
    data_path = '/usr/home/mwemaere/neuro/Data3/'

    criterion = RMSELoss()

    model = resac2()

    test_sat = torch.load(data_path + 'test_ssh_sat.pt')[:,:,:,:88]
    test_transf = torch.load(data_path + 'test_ssh_transf.pt')[:,:,:,:88]
    test_sst = torch.load(data_path + 'test_sst.pt')[:,:,:,:264]
    test_out = torch.load(data_path + 'test_ssh_out.pt')[:,:,:,:264]
    test_u= torch.load(data_path + 'test_u.pt')[:,:,:,:264]
    test_v= torch.load(data_path + 'test_v.pt')[:,:,:,:264]

    test_loader_sat = ConcatData([test_sat,test_sst,test_out,test_u,test_v],shuffle=False)
    test_loader_transformed = ConcatData([test_transf,test_sst,test_out,test_u,test_v],shuffle=False)


    crit = RESACMetric()

    m = crit(test_loader_sat,test_loader_transformed)

    print(m)