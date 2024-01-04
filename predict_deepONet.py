"""
Script for predicting the regression inference of the latent variables by using previously trained MLP and bVAE
"""

import torch 
import numpy as np
import os 
import time

from   utils.configs      import deepOnet_Config as cfg, Make_Name
# Visualisation
import matplotlib.pyplot  as plt 
from   utils.plot         import colorplate as cc 
import utils.plt_rc_setup
from   utils.error        import err, err_z

import tensorflow as tf
from tensorflow.keras import models



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
bVAE_path   =   "05_bVAEs/"

print("#"*30)
print("INFO: Generate Data:")

pred_data_path = save_pred +  case_name +   '.npz'
data = np.load(pred_data_path)

zp_trn = data["zp_trn"]
zp_tst = data["zp_tst"]

gen_params_trn = data["gen_params_trn"]
gen_params_tst = data["gen_params_tst"]

t_trn = gen_params_trn[:,0]
r_trn = gen_params_trn[:,1]

t_tst = gen_params_tst[:,0]
r_tst = gen_params_tst[:,1]

z_trn_out = data["z_trn_out_rescaled"]
z_tst_out = data["z_tst_out_rescaled"]


#Load decoder from bVAE model

decoder_path = bVAE_path + "de_VAE_ld5_b0.002_bs96_ep100_all_r_full_flow.h5"

decoder = models.load_model(decoder_path)

print(f"z_tst_out: {z_tst_out.shape}")
print(f"z_trn_out: {z_trn_out.shape}")
print(f"r_trn, t_trn: {r_trn.shape} ")


#predict
u_pred_tst_out = decoder.predict(z_tst_out)
u_pred_tst_gp = decoder.predict(zp_tst)

u_pred_trn_out = decoder.predict(z_trn_out[:10000])
u_pred_trn_gp =  decoder.predict(zp_trn[:10000])

acc_trn = err(u_pred_trn_out, u_pred_trn_gp)
acc_tst = err(u_pred_tst_out, u_pred_tst_gp)

print(f"acc_trn:{acc_trn}, acc_tst:{acc_tst}")


error_z_trn = err_z(z_trn_out, zp_trn ) *100
error_z_tst = err_z(z_tst_out, zp_tst ) *100

print(f"Error in all latent dimensions for all radii and all time in training set: {error_z_trn}")
print(f"Error in all latent dimensions for all radii and all time in testing set: {error_z_tst}")

r_known = np.unique(r_trn)
r_unknown = np.unique(r_tst)

ld = 5

err_z_r_trn = np.zeros((len(r_known), ld+1))
err_z_r_tst = np.zeros((len(r_unknown), ld+1))

for i, r in enumerate(r_known):
    err_z_r_trn[i,0] = r
    err_z_r_trn[i,1:] = err_z(z_trn_out[(np.where(r_trn == r))], zp_trn[(np.where(r_trn == r))])

for i, r in enumerate(r_unknown):
    err_z_r_tst[i,0] = r
    err_z_r_tst[i,1:] = err_z(z_tst_out[(np.where(r_tst == r))], zp_tst[(np.where(r_tst == r))])


"""
# Creating a figure and axis
fig, ax = plt.subplots()

# Creating the table and adding it to the axis
table = ax.table(cellText=err_z_r_trn, loc='center')

# Modifying the appearance of the table
table.auto_set_font_size(False)
table.set_fontsize(5)
table.scale(1.2, 1.2)  # Adjust the table size if needed

plt.savefig(save_fig + 'mse_z_r_trn.png', dpi= 'figure')


#t = np.linspace(40, 2500, 10)
"""

# Extracting the 'r' column for the x-axis
r_values = err_z_r_trn[:, 0]

# Creating a single plot for all dashed lines
fig, ax = plt.subplots(figsize=(8, 5))

# Loop over the five columns
for i in range(1, 6):  # Start from the second column (index 1)
    z_trn_values = err_z_r_trn[:, i]

    # Plotting the dashed line
    ax.plot(r_values, z_trn_values, linestyle='--', label=f'Z {i}')

    # Plotting a dot with the average value
    #avg_value = np.mean(y_values)
    #ax.scatter(x_values, [avg_value] * len(x_values), marker='o', label=f'Average {i}')

r_values = err_z_r_tst[:,0]
for i in range(1,6):
    z_tst_values = err_z_r_tst[:,i]

    ax.plot(r_values, z_tst_values, 'o', label=f'Z{i}')


# Setting labels and legend
ax.set_xlabel('r')
ax.set_ylabel('Error')
#ax.set_title('Error in Z')
ax.legend()

# Show the plot
plt.savefig(save_fig + 'mse_z_trn.png', dpi = 'figure')
