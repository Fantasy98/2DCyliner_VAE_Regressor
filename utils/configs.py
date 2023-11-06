from torch import nn 
class deepOnet_Config:

    brh_in          =   2  
    brh_out         =   5
    brh_hidden      =   512
    brh_nlayer      =   4
    brh_act         =   nn.Tanh()
    
    trk_in          =   125
    trk_out         =   5
    trk_hidden      =   512
    trk_nlayer      =   4
    trk_act         =   nn.Tanh()


    mrg_in          =   10
    mrg_out         =   5
    mrg_hidden      =   512
    mrg_nlayer      =   4
    mrg_act         =   nn.Tanh()
    
    Epoch           =   100
    lr              =   1e-3 
    batch_size      =   512
    
    train_split     =   0.8
    test_split      =   0.2

    early_stop      =   False
    if early_stop:
        patience    =   20
    else:
        patience    =   0


def Make_Name(cfg):

    """
    Define the name of deepONet case

    Args:
        
        cfg         : The Configuration of DeepOnet

    Returns:

        case_name   : The name of training case    
    """

    case_name   =   f"deepOnet_"+\
                    f"{cfg.brh_in}bin_{cfg.brh_out}bout_{cfg.brh_hidden}bh_{cfg.brh_nlayer}bn_" + \
                    f"{cfg.trk_in}tin_{cfg.trk_out}tout_{cfg.trk_hidden}th_{cfg.trk_nlayer}tn_" + \
                    f"{cfg.trk_in}min_{cfg.trk_out}mout_{cfg.trk_hidden}mh_{cfg.trk_nlayer}mn_" + \
                    f"{cfg.Epoch}epoch_{cfg.batch_size}bs_{int(cfg.train_split*100)}ptrain_"+\
                    f"{cfg.early_stop}ES_{cfg.patience}P"

    return case_name    
                    
