import torch 
import torch.nn as nn

class RH_Penalty(nn.Module):
    """
    Penalty Function based on Relative Humidity used to train the Physics Informed version of the model 
    Calculates RH from predictions and targets of T and q (as well as P) and compares them using MSE
    This penalty is used in the cost function alongside a simple MSE of predictions and targets of T and q
    """
    def __init__(self):
        super(RH_Penalty, self).__init__()
        self.mse = nn.MSELoss()
        
    def RH_norm(self, T_norm, q_norm, P_norm):
        # hard wired values used for normalisation
        mean_T = 271.25368453  
        std_T  = 15.13244937
        mean_q = 0.00368781
        std_q  = 0.00401154 
        mean_P = 69.78005498
        std_P  = 15.99580424
        #T is in K and needs to be in K
        #P is in kPa needs to be in Pa !!!
        T = (T_norm * std_T ) + mean_T
        q = (q_norm * std_q ) + mean_q
        P = (P_norm * std_P ) + mean_P
        P = P * 10**3
        epsilon = 0.622
        esatwat = torch.tensor([611.2]) * torch.exp(17.67*(T-273.15)/(T-273.15+243.5))
        esatice = esatwat * ((T/273.15)**2.66)
        flag = torch.zeros(esatice.shape)
        flag[T<273.15] = 1.0
        esat = (flag*esatice)+((1.0-flag)*esatwat)
        #Convert to qsat
        denom = P-((1-epsilon)*esat)
        qs = epsilon*(esat/denom)
        rh = q/qs
        return rh
    
    def forward(self, predictions, targets, press):
        T_pred = predictions[:,0]
        q_pred = predictions[:,1]
        T_targ = targets[:,0]
        q_targ = targets[:,1]
        #calculate relative humidity 
        rh_pred = self.RH_norm(T_pred, q_pred, press)
        rh_targ = self.RH_norm(T_targ, q_targ, press)
        #calculate penalty
        penalty = self.mse(rh_pred, rh_targ)
        return penalty
    



