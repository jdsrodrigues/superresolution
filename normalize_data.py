import torch
import numpy as np
from scipy import interpolate

def back(X, low_h, inter_h):
    ys = []
    for i in range(X.shape[0]):
        x = X[i]
        f = interpolate.interp1d(low_h, x, kind = "cubic", bounds_error=False, fill_value = (x[0],x[-1]))
        y = f(inter_h)
        ys.append(y)
    fin = np.stack(ys, axis=0)
    return fin

def norm(x, stats):
    mean = stats[0]
    std = stats[1]
    y = (x - mean) / std
    return y

def shuffle(inp, targ, seed=True):
    if seed == True:
        np.random.seed(25)
    shuffler = np.random.permutation(len(inp))
    inp = inp[shuffler] 
    targ = targ[shuffler] 
    return inp, targ, shuffler

def dataset(inp, targ, train_stats=np.zeros(2)):
    if (train_stats == 0.0).all():
        train_stats = np.array([np.mean(inp), np.std(inp)])
        
    inp_norm  = norm(inp, train_stats)
    targ_norm = norm(targ, train_stats)

    inp_new, targ_new, _ = shuffle(inp_norm, targ_norm)
        
    if type(inp_new)!=torch.Tensor:
        inp_new = torch.from_numpy(inp_new)
    if type(targ_new)!=torch.Tensor:
        targ_new = torch.from_numpy(targ_new)
        
    return inp_new, targ_new, train_stats


#----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    PATH = "/data/users/jrodrigu/data_processed/final/"
    # LOW AND ML GRIDS (use only up to 6km - focus on inversions that lead to thin low level clouds)
    low_h = np.load(PATH + "um_h.npy")[:29]
    inter_h = np.load(PATH + "ml_h.npy")[:128]

    stats_T = np.zeros(2)
    stats_q = np.zeros(2)
    stats_P = np.zeros(2)

    for dset in ["TRAIN", "VAL", "TEST"]: 
        print("working on {} dataset".format(dset))
        data_inter = np.load(PATH + "INTER_" + dset + ".npy")[:,:,:128] 
        data_low = np.load(PATH + "LOW_" + dset + ".npy")[:,:,:29] 

        T_inter = data_inter[:,0] + 273.15
        q_inter = data_inter[:,1]
        P_inter = data_inter[:,2]

        T_low = data_low[:,0] + 273.15
        q_low = data_low[:,1]
        P_low = data_low[:,2]

        T_cubic = back(T_low, low_h, inter_h)
        q_cubic = back(q_low, low_h, inter_h)
        P_cubic = back(P_low, low_h, inter_h)

        T_in, T_out, stats_T = dataset(T_cubic, T_inter, stats_T)
        q_in, q_out, stats_q = dataset(q_cubic, q_inter, stats_q)
        P_in, P_out, stats_P = dataset(P_cubic, P_inter, stats_P)

        input_data = torch.stack([T_in,q_in,P_in], axis=1)
        target_data= torch.stack([T_out,q_out,P_out], axis=1)

        data = torch.stack([input_data, target_data], axis=0)
        torch.save(data, PATH + dset + '_data.pt')

    torch.save(stats_T, PATH + 'statsT.pt')
    torch.save(stats_q, PATH + 'statsq.pt')
    torch.save(stats_P, PATH + 'statsP.pt')