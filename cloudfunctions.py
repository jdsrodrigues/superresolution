import torch 

def qsat(T, P):
    
    """ Function to calculate saturation vapour pressure profile from profiles of T and P
        Gives qsat_wat for T>273.15 and qsat_ice for T < 273.15
        T (temperature) needs to be in K
        P (pressure) needs to be in Pa"""

    P = P * 10**3
    epsilon = 0.622
    esatwat = torch.tensor([611.2]) * torch.exp(17.67*(T-273.15)/(T-273.15+243.5))
    esatice = esatwat * ((T/273.15)**2.66)
    flag = torch.zeros(esatice.shape)
    flag[T < 273.15] = 1.0 
    esat = (flag*esatice)+((1.0-flag)*esatwat)
    #Convert to qsat
    denom = P-((1-epsilon)*esat)
    qs = epsilon*(esat/denom)
    
    return qs


def cloud_fraction(qv, qs):
    
    """ Function to calculate cloud fraction from specific humidity and saturation vapour pressure
        qv needs to be in g/g
        qs needs to be in g/g """

    A = 17
    B = 0.95
    qt = qv
    C = 0.5 * (1 + torch.tanh(A * (qt/qs - B)))
    
    return C