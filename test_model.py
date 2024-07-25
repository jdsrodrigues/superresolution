import numpy as np
import torch
from cloudfunctions import qsat, cloud_fraction
from model import SR2CH
                                   
def denorm(x, stats):
    x = x.detach()
    mean = stats[0]
    std = stats[1]
    y = (x * std) + mean
    return y 

# normalization factors
statsT = torch.load('/data/users/jrodrigu/data_processed/final/statsT.pt')
statsq = torch.load('/data/users/jrodrigu/data_processed/final/statsq.pt') 
statsP = torch.load('/data/users/jrodrigu/data_processed/final/statsP.pt') 


# test data
data = torch.load('/data/users/jrodrigu/data_processed/final/TEST_data.pt')

inputs_nn = data[0,:,:2,:]

lowT = denorm(data[0,:,0], statsT)
lowq = denorm(data[0,:,1], statsq)
                   
interT = denorm(data[1,:,0], statsT) 
interq = denorm(data[1,:,1], statsq)
interP = denorm(data[1,:,2], statsP)

print(lowT.shape, interT.shape)
# calculate qsat and CF for truth and cubic data                           
qs_inter = qsat(interT, interP) 
qs_cubic = qsat(lowT, interP) 

C_inter = cloud_fraction(interq, qs_inter)
C_cubic = cloud_fraction(lowq, qs_cubic)
      
# load model weights and biases and make predictions
model = SR2CH().float()
name_model = "NAME"
file = "/data/users/jrodrigu/models/final/" + name_model + ".pth"
      
model.load_state_dict(torch.load(file))
for i, p in enumerate(model.parameters()):
    if i >= 6 and i <=13:
        p = p*0.5
model.eval()
      
predictions = model(inputs_nn.float())
      
predT = denorm(predictions[:,0], statsT)
predq = denorm(predictions[:,1], statsq)      


# calculate qsat and CF from predictions
qs_pred = qsat(predT, interP)
C_pred  = cloud_fraction(predq, qs_pred)

# calculate RMSE and compare cubic and model

criterion = torch.nn.MSELoss()

print("---------------------- cubic ---------------------------")
print("RMSE T: {}".format(criterion(lowT, interT).item() ** 0.5))
print("RMSE q: {}".format(criterion(lowq, interq).item() ** 0.5))
print("RMSE CF: {}".format(criterion(C_cubic, C_inter).item() ** 0.5))

print("---------------------- " + name_model +" ---------------------------")
print("RMSE T: {}".format(criterion(predT, interT).item() ** 0.5))
print("RMSE q: {}".format(criterion(predq, interq).item() ** 0.5))
print("RMSE CF: {}".format(criterion(C_pred, C_inter).item() ** 0.5))