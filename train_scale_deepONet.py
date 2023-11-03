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

data        =   np.load(data_path + "full_genpz_split.npz",mmap_mode="r")
print(f"INFO: Data loaded, keys:")
for k in data.files:
    print(k)

dataset_train = data['dataset_train']
dataset_test  = data['dataset_test']


t_trn     = dataset_train[:,0]
t_trn_max = t_trn.max(); t_trn_min = t_trn.min()
t_trn     = 1 - 2* ((t_trn - t_trn_min)/(t_trn_max - t_trn_min))

r_trn     = dataset_train[:,1]
z_trn_hlp = dataset_train[:,2:-5]

z_trn_out = dataset_train[:, -5: ]

t_tst     = dataset_test[:,0]
t_tst_max = t_tst.max(); t_tst_min = t_tst.min()
t_tst     = 1 - 2* ((t_tst - t_tst_min)/(t_tst_max - t_tst_min))


r_tst     = dataset_test[:,1]
z_tst_hlp = dataset_test[:,2:-5]
z_tst_out = dataset_test[:,-5:]

X         = (torch.from_numpy(dataset_train[:,:2]),torch.from_numpy(z_trn_hlp))
y         = torch.from_numpy(z_trn_out)


train_dl, val_dl = make_DataLoader(X,y,batch_size=cfg.batch_size,train_split=cfg.train_split)

del X , y
print(f"INFO: DataLoader generated, NTrain={len(train_dl)}, Nval={len(val_dl)}")


print("#"*30)
print(f"Compile Model")

model       = DeepONet( cfg.brh_in, cfg.brh_out, cfg.brh_hidden, cfg.brh_nlayer, cfg.brh_act, 
                        cfg.trk_in, cfg.trk_out, cfg.trk_hidden, cfg.trk_nlayer, cfg.trk_act,
                        cfg.mrg_in, cfg.mrg_out, cfg.mrg_hidden, cfg.mrg_nlayer, cfg.mrg_act,
                        )

optimizer   = torch.optim.Adam(model.parameters(), lr = cfg.lr, eps= 1e-7)

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
print("Validating On Training Data")



z_sample_p  = model(torch.from_numpy(dataset_train[:,:2]).float(), torch.from_numpy(z_trn_hlp).float()).detach().cpu().numpy()


for i in range(z_tst_out.shape[-1]):
    fig, axs = plt.subplots(1,1)
    axs.plot(dataset_train[:,1], z_trn_out[:,i], ".", c = cc.black)
    axs.plot(dataset_train[:,1], z_sample_p[:,i], ".", c = cc.red)
    axs.set_xlabel("Radius")
    axs.set_ylabel(rf"$z_{i}$")
    plt.legend(["Reference", "Prediction"])
    plt.savefig(save_fig + f"Train_Z_{i+1}_" +  case_name + ".jpg", bbox_inches='tight')


print("#"*30)
print("Validating On test data")

z_sample_p  = model(torch.from_numpy(dataset_test[:,:2]).float(), torch.from_numpy(z_tst_hlp).float()).detach().cpu().numpy()

np.savez_compressed(save_pred + case_name + '.npz',
                    zp  = z_sample_p,)
                 
for i in range(z_tst_out.shape[-1]):
    fig, axs = plt.subplots(1,1)
    axs.plot(dataset_test[:,1], z_tst_out[:,i], ".", c = cc.black)
    axs.plot(dataset_test[:,1], z_sample_p[:,i], "x", c = cc.red)
    axs.set_xlabel("Radius")
    axs.set_ylabel(rf"$z_{i+1}$")
    plt.legend(["Reference", "Prediction"])
    plt.savefig(save_fig + f"Test_Z_{i+1}_" +  case_name + ".jpg", bbox_inches='tight')