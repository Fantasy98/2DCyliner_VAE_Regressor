"""
Model for Implementing the DeepONet for regression problem. 
I would like to use random sample as a part of input 
"""

import torch 
from   torch import nn 
import torch.nn.functional as F
import numpy as np 
import math
class DeepONet(nn.Module):
    def __init__(self, 
                 brh_in, brh_out, brh_hidden, brh_nlayer, brh_act, 
                 trk_in, trk_out, trk_hidden, trk_nlayer, trk_act, 
                 mrg_in, mrg_out, mrg_hidden, mrg_nlayer, mrg_act, 
                 
                 ):
        """
        A deepONet-like solver for the buliding the relation between physcial information and latent variables


        Args: 
        
            brh_in      :  (int) Input of the branch Net (r,t)
            
            brh_out     :  (int) Output of the branch Net
            
            brh_hidden  :  (int) Hidden size of the branch Net
            
            brh_nlayer  :  (int) Number of layer of the branch Net
            
            brh_act     :  (nn.) Activation Func of the branch Net


            trk_in      :  (int) Input of the trunk Net (z_hlp)
            
            trk_out     :  (int) Output of the trunk Net
            
            trk_hidden  :  (int) Hidden size of the trunk Net
            
            trk_nlayer  :  (int) Number of layer of the trunk Net
            
            trk_act     :  (nn.) Activation Func of the trunk Net


            mrg_in      :  (int) Input of the merge Net
            
            mrg_out     :  (int) Output of the merge Net
            
            mrg_hidden  :  (int) Hidden size of the merge Net
            
            mrg_nlayer  :  (int) Number of layer of the merge Net
            
            mrg_act     :  (nn.) Activation Func of the merge Net


        """
        super(DeepONet,self).__init__()

        self.branchNet  =   self.Build_Branch_Net(brh_in, brh_out, brh_hidden, brh_nlayer, brh_act)
        self.trunkNet   =   self.Build_Trunk_Net(trk_in, trk_out, trk_hidden, trk_nlayer, trk_act)
        self.mrgNet     =   self.Build_Merge_Net(mrg_in, mrg_out, mrg_hidden, mrg_nlayer, mrg_act)
        self.brh_out    =   brh_out
        self.trk_out    =   trk_out
        self.mrg_in     =   mrg_in
        
        self.W_q = nn.Linear(trk_out, trk_out)
        self.W_k = nn.Linear(trk_out, trk_out)
        self.W_v = nn.Linear(brh_out, brh_out)
        self.W_o = nn.Linear(brh_out, brh_out)

        self.Conv = nn.Conv1d(2,1,kernel_size=(1,2))


    def Build_Branch_Net(self, brh_in, brh_out, brh_hidden, brh_nlayer, brh_act:nn.Tanh()):
        """
        The function for bulidng the Network of branch: 
            
        Args: 

            brh_in      :  (int) Input of the branch Net

            brh_out     :  (int) Output of the branch Net
            
            brh_hidden  :  (int) Hidden size of the branch Net
            
            brh_nlayer  :  (int) Number of layer of the branch Net
            
            brh_act     :  (nn.module) Activation Func of the branch Net
        
        Returns:

            model       :   (nn.Module) Architecture for the Branch Net
        """

        
        model = nn.Sequential()

        ## Build input 
        layer   =   nn.Linear(brh_in, brh_hidden);
        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        model.add_module("brh_in", layer)
        
        ## Build hidden layers
        for i in range(brh_nlayer):
            layer   =   nn.Linear(brh_hidden, brh_hidden);
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)    
            model.add_module(f"brh_hid_{i+1}", layer)
            model.add_module(f"brh_act_{i+1}", brh_act)
        
        ## Build output
        layer   =   nn.Linear(brh_hidden,brh_out);
        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        model.add_module("brh_out", layer)

        print(f"INFO: Branch Built, Architecture: \n{model.eval}")
        return model 

    def Build_Trunk_Net(self, trk_in, trk_out, trk_hidden, trk_nlayer, trk_act:nn.Tanh()):
        """
        The function for bulidng the Network of Trunk : 
            
        Args: 

            trk_in      :  (int) Input of the Trunk  Net

            trk_out     :  (int) Output of the Trunk  Net
            
            trk_hidden  :  (int) Hidden size of the Trunk  Net
            
            trk_nlayer  :  (int) Number of layer of the Trunk  Net
            
            trk_act     :  (nn.module) Activation Func of the Trunk  Net
        
        Returns:

            model       :   (nn.Module) Architecture for the Trunk  Net
        """

        
        model = nn.Sequential()

        ## Build input 
        layer   =   nn.Linear(trk_in, trk_hidden);   #Q is new layer added to olde later? add module?
        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        model.add_module("trk_in", layer)
        
        ## Build hidden layers
        for i in range(trk_nlayer):
            layer   =   nn.Linear(trk_hidden, trk_hidden);
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)    
            model.add_module(f"trk_hid_{i+1}", layer)
            model.add_module(f"trk_act_{i+1}", trk_act)
        
        ## Build output
        layer   =   nn.Linear(trk_hidden,trk_out);
        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        model.add_module("trk_out", layer)

        print(f"INFO: Trunk Built, Architecture: \n{model.eval}")
        return model
    
    def Build_Merge_Net(self, mrg_in, mrg_out, mrg_hidden, mrg_nlayer, mrg_act:nn.Tanh()):
        """
        The function for bulidng the Network of Trunk : 
            
        Args: 

            mrg_in      :  (int) Input of the Trunk  Net

            mrg_out     :  (int) Output of the Trunk  Net
            
            mrg_hidden  :  (int) Hidden size of the Trunk  Net
            
            mrg_nlayer  :  (int) Number of layer of the Trunk  Net
            
            mrg_act     :  (nn.module) Activation Func of the Trunk  Net
        
        Returns:

            model       :   (nn.Module) Architecture for the Trunk  Net
        """

        
        model = nn.Sequential()

        ## Build input 
        layer   =   nn.Linear(mrg_in, mrg_hidden);
        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        model.add_module("mrg_in", layer)
        
        ## Build hidden layers
        for i in range(mrg_nlayer):
            layer   =   nn.Linear(mrg_hidden, mrg_hidden);
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)    
            model.add_module(f"mrg_hid_{i+1}", layer)
            model.add_module(f"mrg_act_{i+1}", mrg_act)
        
        ## Build output
        layer   =   nn.Linear(mrg_hidden,mrg_out);
        nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        model.add_module("mrg_out", layer)

        print(f"INFO: Trunk Built, Architecture: \n{model.eval}")
        return model

    def Attn_Operator(self, yB, yT):
        """
        We use the trunk and branch output for computing the attention 

        Args:

            yB  :   The ouput from branch Net 
            yT  :   The ouput from trunk Net 
        
        Returns:

            The attention product as the prediction
        """
        Q = self.W_q(yT)
        K = self.W_k(yT)
        V = self.W_v(yB)

        x = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.trk_out)
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, V)
        x = self.W_o(x)
        x  = self.Conv(x)
        x = yB * yT
        return F.tanh(x)   

    def Concat_Operator(self, yB, yT):
        """
        We use the trunk and branch output for computing the attention 

        Args:

            yB  :   The ouput from branch Net 
            yT  :   The ouput from trunk Net 
        
        Returns:

            concatenating the outputs of two nets  (bs, N_yB + N_yT )
        """


        x  = torch.concat([yB,yT],dim=1)


        return x

    def Dot_Operator(self, yB, yT):
        """
        We use the trunk and branch output for computing the attention 

        Args:

            yB  :   The ouput from branch Net 
            yT  :   The ouput from trunk Net 
        
        Returns:

            dot product of two net outputs (bs,1)
        """
 
        x = torch.zeros(len(yT),1)
        for i in range(len(yT)):             #or len(yB) should be same as batch size
            
            #print(f"shape of yT:{yT[i].shape}, yB:{yB[i].shape}")
            x[i] = torch.dot(yB[i],yT[i])
        
        device      = ("cuda:0" if torch.cuda.is_available() else "cpu" )

        if device == "cuda:0":
            x = x.to(device)
        #print(f"Device of gpu_tensor yT:{yT.device}, yB:{yB.device}, x:{x.device}")

        return x


    def elementwise_multiply(self, yB, yT, window_size):
        result = []
        for i in range(yB.size(0) - window_size + 1):
            window1 = yB[i:i+window_size]
            window2 = yT[i:i+window_size]
            result.append(torch.mul(window1, window2))
        
        x = torch.stack(result)
        #print(f"shape of x:{x.shape}, x0:{x[0].shape}, device: {x.device}")
        return x

    def sliding_window_multiply(self, yB, yT):

        yB_np = yB.detach().cpu().numpy()
        yT_np = yT.detach().cpu().numpy()

        #print(f"shape of yB_np:{yB_np.shape}, yb0: {yB_np[0]}")
        x = torch.zeros(len(yT), self.mrg_in)

        for i in range(len(yT)):                                                     #or len(yB) should be same as batch size
            
            #print(f"shape of yT:{yT[i].shape}, yB:{yB[i].shape}")
            x[i] = torch.from_numpy(np.convolve( yB_np[i] , yT_np[i] ))
        

        device      = ("cuda:0" if torch.cuda.is_available() else "cpu" )

        if device == "cuda:0":
            x = x.to(device)
    
        #print(f"shape of x:{x.shape}, x0:{x[0].shape}, device: {x.device}")
        return x

    def forward(self,xB,xT):
        """"
        Feed-forward Propagation

        Args:
            xB  :   Input for branch
            xT  :   Input for trunk
        
        Returns:
            
            The inference from model 
        """
        xB = self.branchNet(xB)
    
        xT = self.trunkNet(xT)

        x =  self.sliding_window_multiply(xB, xT)

        return self.mrgNet(x)
    

if __name__ == "__main__":
    brh_in=5; brh_out=5; brh_hidden=20; brh_nlayer=2; brh_act=nn.Tanh() 
    trk_in=2; trk_out=5; trk_hidden=20; trk_nlayer=2; trk_act=nn.Tanh() 
    
    xB = torch.randn(size=(1,2))
    xT = torch.randn(size=(1,5))

    model = DeepONet(brh_in, brh_out, brh_hidden, brh_nlayer, brh_act, 
                    trk_in, trk_out, trk_hidden, trk_nlayer, trk_act, )

    y = model(xB,xT)
    print(f"Forward done, Output has shape = {y.shape}")