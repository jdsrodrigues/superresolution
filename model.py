import torch
from torch import nn
from pixshuf import PixelShuffle1D

class SR2CH(nn.Module):
    def __init__(self, ch = 2, ch1 = 32, ch2 = 64, ch3 = 128, ch4 = 256): 
        super(SR2CH, self).__init__()
        #DOWN
        self.d1 = nn.Conv1d(ch, ch1, kernel_size = 3, stride = 2, padding = 3//2, padding_mode='reflect')
        self.d2 = nn.Conv1d(ch1, ch2, kernel_size = 3, stride = 2, padding = 3//2, padding_mode='reflect')
        self.d3 = nn.Conv1d(ch2, ch3, kernel_size = 3, stride = 2, padding = 3//2, padding_mode='reflect')
        #BOTTLENECK
        self.b = nn.Conv1d(ch3, ch3, kernel_size = 3, stride = 2, padding = 3//2, padding_mode='reflect')
        #UP
        self.u3 = nn.Conv1d(ch3, ch4, kernel_size = 3, padding = 3//2, padding_mode='reflect')
        self.u2 = nn.Conv1d(ch4, ch3, kernel_size = 3, padding = 3//2, padding_mode='reflect')
        self.u1 = nn.Conv1d(ch3, ch2, kernel_size = 3, padding = 3//2, padding_mode='reflect')
        #FINAL
        self.f = nn.Conv1d(ch2, ch*2, kernel_size = 3, padding = 3//2, padding_mode='reflect')

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.2)
        
        self.ps = PixelShuffle1D(2)
        
        self.drop  = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(ch1) 
        self.norm2 = nn.BatchNorm1d(ch2)
        self.norm3 = nn.BatchNorm1d(ch3)
        self.norm4 = nn.BatchNorm1d(ch4)
        
    def forward(self, x):
        X = x
        #DOWN
        x = self.lrelu(self.d1(x))
        x1 = x                        
        x = self.lrelu(self.d2(x))
        x2 = x                   
        x = self.lrelu(self.d3(x))
        x3 = x     
        #BOTTLENECK
        x = self.relu(self.drop(self.b(x)))  
        #UP
        x = self.relu(self.norm4(self.drop(self.u3(x)))) 
        x = self.ps(x)   
        x = torch.cat((x, x3), 1) 
        x = self.relu(self.norm3(self.drop(self.u2(x))))   
        x = self.ps(x)                     
        x = torch.cat((x, x2), 1)            
        x = self.relu(self.norm2(self.drop(self.u1(x)))) 
        x = self.ps(x)                      
        x = torch.cat((x, x1), 1)              
        #FINAL
        x = self.f(x)                            
        x = self.ps(x)                       
        x = x + X                              
        return x
