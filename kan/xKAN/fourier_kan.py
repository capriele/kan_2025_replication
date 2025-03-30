import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *

# code modified from https://github.com/GistNoesis/FourierKAN/

#This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
#It should be easier to optimize as fourier are more dense than spline (global vs local)
#Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
#The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
#Avoiding the issues of going out of grid

class FourierKANLayer(nn.Module):
    def __init__(self, 
                 in_dim=3, 
                 out_dim=2, 
                 num=5,
                 k=3, 
                 noise_scale=0.5, 
                 scale_base_mu=0.0, 
                 scale_base_sigma=1.0, 
                 scale_sp=1.0, 
                 base_fun=torch.nn.SiLU(), 
                 grid_eps=0.02, 
                 grid_range=[-1, 1], 
                 sp_trainable=True, 
                 sb_trainable=True, 
                 save_plot_data = True, 
                 device='cpu', 
                 sparse_init=False,
                 addbias=True):
        super(FourierKANLayer,self).__init__()
        self.num = num
        self.addbias = addbias
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2,out_dim,in_dim,num) / 
                                             (np.sqrt(in_dim) * np.sqrt(self.num) ) )
        if self.addbias:
            self.bias  = torch.nn.Parameter(torch.zeros(1,out_dim))
        
        self.to(device)

    
    def to(self, device):
        super(FourierKANLayer, self).to(device)
        self.device = device    
        return self

    #x.shape ( ... , indim ) 
    #out.shape ( ..., out_dim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.out_dim,)
        x = torch.reshape(x,(-1,self.in_dim))
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1,self.num+1,device=x.device),(1,1,1,self.num))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1))
        print(xrshp)
        #This should be fused to avoid materializing memory
        c = torch.cos(k*xrshp)
        s = torch.sin(k*xrshp)
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  torch.sum(c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += torch.sum(s*self.fouriercoeffs[1:2],(-2,-1))

        print(self.fouriercoeffs[0:1])
        print(self.fouriercoeffs[1:2])

        if(self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape(y, outshape)
        return y

class Fourier_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8,
        spline_order: int = 0, #  placeholder
        device = 'cpu',
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FourierKANLayer(
                in_dim=in_dim, 
                out_dim=out_dim,
                num=grid_size,
                device = device,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
