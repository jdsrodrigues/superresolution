import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from normalize_data import shuffle
from penalty_function import RH_Penalty
from model import SR2CH

def plot_learming_curve(loss_train,loss_val, name_model):
    ep = loss_train.keys()
    tl = loss_train.values()
    vl = loss_val.values()
    plt.figure()
    plt.plot(ep, tl, linestyle="--", label = "train loss")
    plt.plot(ep, vl, label = "validation loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Learning Curve - {}".format(name_model))
    plt.savefig("learning_curve_{}".format(name_model))
    
def training_net(model, data, pressure, criterion, penalty_func, optimizer, alpha=0.0, num_epochs=100, batch_size=64, early_th=10):
    train_loss_dict = {}
    val_loss_dict = {}
    best_epoch = 0
    best_loss = 100000
    worse = 0
    # Data
    train_set, val_set = data
    train_inputs, train_targets = train_set
    val_inputs, val_targets = val_set
    train_press, val_press = pressure
    
    for epoch in range(1, num_epochs+1):
        train_inputs, train_targets, shuffler = shuffle(train_inputs, train_targets, seed=False)
        train_press = train_press[shuffler]
        batch_idxs = int(train_inputs.shape[0] // batch_size)
        model.train()
        for idx in range(batch_idxs):       
            batch_inputs   = train_inputs[idx*batch_size : (idx+1)*batch_size]
            batch_targets  = train_targets[idx*batch_size : (idx+1)*batch_size]
            batch_press    = train_press[idx*batch_size : (idx+1)*batch_size]
            optimizer.zero_grad()
            batch_pred = model(batch_inputs.float())
            batch_mse = criterion(batch_pred.float(), batch_targets.float())
            batch_penalty = penalty_func(batch_pred, batch_targets, batch_press)
            train_loss = (1-alpha) * batch_mse +  alpha * batch_penalty
            train_loss.backward()
            optimizer.step()
   
        model.eval()
        val_pred = model(val_inputs.float())
        val_mse = criterion(val_pred.float(), val_targets.float())
        val_penalty = penalty_func(val_pred.float(), val_targets.float(), val_press)
        val_loss = (1-alpha) * val_mse +  alpha * val_penalty
        print('Epoch {}:'.format(epoch))
        print("Train Loss = {}".format(train_loss))
        print("Validation Loss = {}".format(val_loss))
        
        train_loss_dict[epoch] = train_loss.item()
        val_loss_dict[epoch] = val_loss.item()
        
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss.item()
            best_weights = model.state_dict()
            worse=0
        else:
            worse+=1
            
        if worse==early_th:
            break
        
    print('Best epoch : {}'.format(best_epoch))
    print('Best val loss : {}'.format(best_loss))
    

    return best_weights, train_loss_dict, val_loss_dict

#---------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    name_model = "NAME"

    #alpha=0.0 for traditional ml
    #alpha>0.0 for physics informed ml

    alpha    = 0.1
    lr_rate  = 1e-4
    n_epochs = 100
    b_size   = 64 

    model = SR2CH().float()

    criterion = nn.MSELoss()
    penalty_func = RH_Penalty()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr_rate) 

    traindata = torch.load('/data/users/jrodrigu/data_processed/final/TRAIN_data.pt')
    valdata   = torch.load('/data/users/jrodrigu/data_processed/final/VAL_data.pt')

    P_train = traindata[1, :, 2, :] 
    P_val   = valdata[1, :, 2, :]   

    trainset = traindata[:, :, :2, :] 
    valset   = valdata[ :, :, :2, :]

    pressure = [P_train, P_val]
    data = [trainset, valset]

    start = time.time()

    final_weights, losses_train, losses_val = training_net(model, data, pressure, criterion, penalty_func, optimizer, alpha=alpha) 

    end = time.time()

    print("End of Training - took {}".format(end-start))
    
    plot_learming_curve(losses_train, losses_val, name_model)

    with open('/data/users/jrodrigu/losses/final/train_loss_{}.pkl'.format(name_model), 'wb') as fp:
        pickle.dump(losses_train, fp)
    with open('/data/users/jrodrigu/losses/final/val_loss_{}.pkl'.format(name_model), 'wb') as fp:
        pickle.dump(losses_val, fp)

    torch.save(final_weights, '/data/users/jrodrigu/models/final/{}.pth'.format(name_model))

    print("Files Saved - Losses and Weights")