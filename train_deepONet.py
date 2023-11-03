"""
Script for training a network for generate the regression inference of the latent variables 
"""

import torch 
import numpy as np
import os 
import time
from   utils.NNs.DeepONet import DeepONet
from   utils.train        import fit
from   utils.datas        import make_DataLoader
from   utils.configs      import deepOnet_Config as cfg, Make_Name
# Visualisation
import matplotlib.pyplot  as plt 
from   utils.plot         import colorplate as cc 
import utils.plt_rc_setup
##

np.random.seed(1024)
torch.manual_seed(1024)
device      = ("cuda:0" if torch.cuda.is_available() else "cpu" )
print(f"INFO: The training will be undertaken on {device}")

case_name   = Make_Name(cfg)
print(f"CASE NAME:\n{case_name}")  

data_path   =   "01_Data/"
save_pred   =   "02_Pred/"
save_model  =   "03_CheckPoint/"
save_fig    =   "04_Figs/"
print("#"*30)
print("INFO: Generate Data:")

d           =   np.load(data_path + "full_genpz.npz",mmap_mode="r")
print(f"INFO: Data loaded, keys:")
for k in d.files:
    print(k)

genPara     =   d["gen_params"]
zsample     =   d["z"]


# zsample     =   1 - 2*((zsample - zsample.min())/(zsample.max()- zsample.min()))
print(f"Total Data: {zsample.shape}")
num_test    =   int(len(genPara) * (1-cfg.test_split))
train_mask  =   np.random.choice(len(genPara), num_test, replace=False)

Train_Mask  =   np.zeros(shape=genPara.shape[0],dtype=np.bool8)
Train_Mask[train_mask]  =   1
Test_Mask   =   np.ones(shape=genPara.shape[0],dtype=np.bool8)
Test_Mask[train_mask]   =   0
print(f"There are {np.sum(Train_Mask)} for training")

test_genPara = genPara[Test_Mask,:2]
test_zsample = zsample[Test_Mask,:]

genPara     =   genPara[Train_Mask,:]
zsample     =   zsample[Train_Mask,:]
print(f"The Generation parameters has shape ={genPara.shape}")
print(f"The Sampled Latent variabels has shape ={zsample.shape}")

TimeFactor  =   genPara[:,0]
TimeFactor  =   1- 2*((TimeFactor - TimeFactor.min())/(TimeFactor.max()-TimeFactor.min()))
GeoFactor   =   genPara[:,1]

Factors     =   np.stack([TimeFactor,GeoFactor],axis=-1)


X           = (torch.from_numpy(Factors),torch.from_numpy(Factors))
y           = torch.from_numpy(zsample)

del TimeFactor, GeoFactor
train_dl, val_dl = make_DataLoader(X,y,batch_size=cfg.batch_size,train_split=cfg.train_split)

del X , y
print(f"INFO: DataLoader generated, NTrain={len(train_dl)}, Nval={len(val_dl)}")


print("#"*30)
print(f"Compile Model")

model       = DeepONet(cfg.brh_in, cfg.brh_out, cfg.brh_hidden, cfg.brh_nlayer, cfg.brh_act, 
                 cfg.trk_in, cfg.trk_out, cfg.trk_hidden, cfg.trk_nlayer, cfg.trk_act,
                 cfg.mrg_in, cfg.mrg_out, cfg.mrg_hidden, cfg.mrg_nlayer, cfg.mrg_act,
                 
                 )

optimizer   = torch.optim.SGD(model.parameters(), lr = cfg.lr, momentum=0.98)

loss_fn     = torch.nn.MSELoss()

print(f"INFO: The Model has been compiled!")

print("#"*30)
print(f"Training")
start_time  = time.time()

hist        = fit(device,
                  model,
                  train_dl,loss_fn,
                  cfg.Epoch,
                  optimizer,
                  val_dl,
                  if_early_stop=cfg.early_stop,
                  patience=cfg.patience)

end_time  = time.time()
cost_time = end_time - start_time

print(f"Training Ended, Cost Time = {cost_time:.2f}s")

print("#"*30)
print("Save Checkpoint")
ckpt = {     "model":model.state_dict(),
             "history":hist,
             "time":cost_time}
torch.save( ckpt,
            save_model + case_name + ".pt"  )

print(f"INFO: CheckPoint saved!")

print("#"*30)
print("Validating")
InputRandom =   np.random.random(size=(len(test_genPara), zsample.shape[-1]))

z_sample_p  = model(torch.from_numpy(test_genPara).float(), torch.from_numpy(test_genPara).float()).detach().cpu().numpy()
error       = np.linalg.norm((z_sample_p - test_zsample))/np.linalg.norm(test_zsample)

print(f"l2-norm of error: {error}")
np.savez_compressed(save_pred + case_name + '.npz',
                    zt  = test_zsample,
                    zp  = z_sample_p,
                    xt  = test_genPara)

for i in range(test_zsample.shape[-1]):
    fig, axs = plt.subplots(1,1)
    axs.plot(test_zsample[:,i], ".", c = cc.black)
    axs.plot(z_sample_p[:,i], ".", c = cc.red)
    plt.savefig(save_fig + f"Z_{i+1}_" +  case_name + ".jpg", bbox_inches='tight')