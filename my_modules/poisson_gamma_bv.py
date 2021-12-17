# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:48:27 2021

@author: yokoyama
"""
import numpy as np
from scipy.special import psi

# references:
# [1]  
# [2] 
# [3] https://github.com/msamunetogetoge/masamune/blob/master/variation_inference_blog.ipynb

class VI():
    def __init__(self,x,a,b,pi):
        self.x=x
        self.a = a
        self.b =b
        self.N = len(self.x)
        self.pi = np.zeros(self.N)
        self.Tau =np.array([])

    def Estep(self):
        self.E_lam = self.a/self.b
        self.E_loglam = psi(self.a) -np.log(self.b)
        log_pi = np.zeros(self.N)
        
        if len(self.Tau)==0:
            for i in range(self.N):
                log_pi[i] =  np.sum(self.x[:i] *self.E_loglam[0]  - self.E_lam[0] ) + np.sum(self.x[i:]*self.E_loglam[1] - self.E_lam[1]  )
        else:
            tau = int(self.Tau[-1])
            for i in range(0, tau):
                log_pi[i] =  np.sum(self.x[:i] *self.E_loglam[0]  - self.E_lam[0] )
                
            for i in range(tau, self.N):
                log_pi[i] =  np.sum(self.x[i:] *self.E_loglam[1]  - self.E_lam[1] )
                # log_pi[i] =  np.sum(self.x[:i] *self.E_loglam[0]  - self.E_lam[0] ) + np.sum(self.x[i:] *self.E_loglam[1]  - self.E_lam[1] )
        
        # for i in range(self.N):
        #     log_pi[i] =  np.sum(self.x[:i] *self.E_loglam[0]  - self.E_lam[0] ) + np.sum(self.x[i:]*self.E_loglam[1] - self.E_lam[1]  )
        
        log_pi-= np.max(log_pi)
        self.pi = np.exp(log_pi)
        self.pi /= np.sum(self.pi)    


        return  self.pi

    def Mstep( self):
        pi       = self.Estep()
        tau      = np.argmax(pi)
        
        delta    = np.zeros((2,self.N))
        delta[0, :tau]   = 1
        delta[1, tau+1:] = 1
        
        self.E_d = np.zeros((2,self.N))
        
        for i in range(0, tau):
            self.E_d[0,i] = np.sum(pi[:i] * delta[0,:i])
            
        for i in range(tau,self.N):
            self.E_d[1,i] = np.sum(pi[i:] * delta[1,i:])
        
        # for i in range(self.N):  
        #     self.E_d[0,i] = np.sum(pi[i+1:])
        #     self.E_d[1,i] = np.sum(pi[:i+1])

        a_hat =np.dot(self.x, self.E_d.T ) + self.a
        b_hat = np.sum(self.E_d, axis=1) +self.b

        return  a_hat , b_hat

    
    def itr_calc(self,max_itr ):
        
        for i in range(max_itr):
            self.a,self.b = self.Mstep()
            self.pi = self.Estep()
            tau = np.argmax(self.pi)
            self.Tau =np.append(self.Tau,tau)
        
        E_x =np.zeros(self.N)
        for i in range( self.N):
            if i< tau:
                E_x[i] = (self.a/self.b)[0]
            else :
                E_x[i] = (self.a/self.b)[1]     


        return self.pi, E_x