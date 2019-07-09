#!/usr/bin/env python
# coding: utf-8

# ##Algorithms for gradient descent in two dimensions

# ## Import libraries

# In[ ]:


import numpy as np
import math


# ## Gradient descent algorithms
# 
# First:
# * Choose starting points for X1 and X2
# * Find the gradients at the starting point
# * If the Euclidean norm of the two-dimentional derivatives of loss function is close to zero then stop
# * Otherwise iterate
# 
# Iterate as follows:
# * Take small steps in the downhill direction (opposite to the gradient)
# * Find the gradients
# * If the Euclidean norm of the two-dimentional derivatives of loss function is close to zero then stop

# In[ ]:


class gd_2d:                                          #define a class gd_2d
    
    def __init__(self, fn_loss, fn_grad1, fn_grad2):  #using the __init__ to initialize the coding class
        self.fn_loss = fn_loss                        #involve 'self' to pass the method a reference to the parent class
        self.fn_grad1 = fn_grad1
        self.fn_grad2 = fn_grad2
        
    def pv(self, x1_init, x2_init, n_iter, eta, tol, tol_upper):  #define plain vanilla gradient descent method
        
        x1 = x1_init                                  #define variable x1
        x2 = x2_init                                  #define variable x2
        
        loss_path = []                                #initialise lists to score the path of x1, x2, and the path of the loss function
        x1_path = []
        x2_path = []
        
        x1_path.append(x1)
        x2_path.append(x2)
        
        loss_this = self.fn_loss(x1, x2)
        loss_path.append(loss_this)
        g1 = self.fn_grad1(x1, x2)
        g2 = self.fn_grad2(x1, x2)
        
        for i in range(n_iter):
            if  math.sqrt(g1**2 + g2**2) < tol or loss_this > tol_upper:
                break
            g1 = self.fn_grad1(x1, x2)
            g2 = self.fn_grad2(x1, x2)
            x1 += -eta * g1
            x1_path.append(x1)
            x2 += -eta * g2
            x2_path.append(x2)
            loss_this = self.fn_loss(x1, x2)
            loss_path.append(loss_this)
            
        if loss_this > tol_upper:
            print('Exploded')
        elif math.sqrt(g1**2 + g2**2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x1 = {} x2 = {}'.format(i, loss_this, x1, x2))
        self.loss_path = np.array(loss_path)
        self.x1_path = np.array(x1_path)
        self.x2_path = np.array(x2_path)
        
    def momentum(self, x1_init, x2_init, n_iter, eta, tol, tol_upper, alpha):
   
        x1 = x1_init
        x2 = x2_init
        
        loss_path = []
        x1_path = []
        x2_path = []
        nu1_path = []
        
        x1_path.append(x1)
        x2_path.append(x2)
        loss_this = self.fn_loss(x1, x2)
        loss_path.append(loss_this)
        g1 = self.fn_grad1(x1, x2)
        g2 = self.fn_grad2(x1, x2)
        nu1 = 0
        nu1_path.append(nu1)
        nu2 = 0
        
        for i in range(n_iter):
            g1 = self.fn_grad1(x1, x2)
            g2 = self.fn_grad2(x1, x2)
            
            if math.sqrt(g1**2 + g2**2) < tol or loss_this > tol_upper:
                break

            nu1 = alpha * nu1 + eta * g1
            nu1_path.append(nu1)
            nu2 = alpha * nu2 + eta * g2
            x1 += -nu1
            x1_path.append(x1)
            x2 += -nu2
            x2_path.append(x2)
            loss_this = self.fn_loss(x1, x2)
            loss_path.append(loss_this)

        if loss_this > tol_upper:
            print('Exploded')
        elif math.sqrt(g1**2 + g2**2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x1 = {} x2 = {}'.format(i, loss_this, x1, x2))
        self.loss_path = np.array(loss_path)
        self.x1_path = np.array(x1_path)
        self.x2_path = np.array(x2_path)
    
    def nag(self, x1_init, x2_init, n_iter, eta, tol, tol_upper, alpha):
        x1 = x1_init
        x2 = x2_init
        
        loss_path = []
        x1_path = []
        x2_path = []
        
        x1_path.append(x1)
        x2_path.append(x2)
        loss_this = self.fn_loss(x1, x2)
        loss_path.append(loss_this)
        g1 = self.fn_grad1(x1, x2)
        g2 = self.fn_grad2(x1, x2)
        nu1 = 0
        nu2 = 0

        for i in range(n_iter):
            # i starts from 0 so add 1
            # The formula for mu was mentioned by David Barber UCL as being Nesterovs suggestion
            mu = 1 - 3 / (i + 1 + 5) 
            g1 = self.fn_grad1(x1 - mu*nu1, x2 - mu*nu2)
            g2 = self.fn_grad2(x1 - mu*nu1, x2 - mu*nu2)
            
            if math.sqrt(g1**2 + g2**2) < tol or loss_this > tol_upper:
                break

            nu1 = alpha * nu1 + eta * g1
            nu2 = alpha * nu2 + eta * g2
            x1 += -nu1
            x1_path.append(x1)
            x2 += -nu2
            x2_path.append(x2)
            loss_this = self.fn_loss(x1, x2)
            loss_path.append(loss_this)
            
        if loss_this > tol_upper:
            print('Exploded')
        elif math.sqrt(g1**2 + g2**2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by x1 = {} x2 = {}'.format(i, loss_this, x1, x2))
        self.loss_path = np.array(loss_path)
        self.x1_path = np.array(x1_path)
        self.x2_path = np.array(x2_path)

