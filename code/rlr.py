import numpy as np
import scipy.special 
from scipy.optimize import minimize

class rlr:

    def __init__(self):
        pass
             
    def discriminant(self,theta,X):
        #Complete this implementation
        X_absorbed = np.hstack((X,np.ones((X.shape[0],1))))
        res = X_absorbed@theta
        return res.reshape(-1,1)
    
    def predict(self,theta,X):
        #Complete this implementation
        X = np.hstack((X,np.ones((X.shape[0],1))))
        return np.sign(X@theta).reshape(-1,1)

    def risk(self,theta,X,Y):
        #Complete this implementation
        discriminant_x = self.discriminant(theta,X)
        risk = np.sum(np.log(1+np.exp(-Y*discriminant_x)))
        return risk
        
    def regularizer(self,theta):
        #Complete this implementation
        epsilon = 1e-2
        theta = theta[:-1]
        return np.sum(np.sqrt(theta**2+epsilon))
        
    def regularized_risk(self,theta,X,Y,lam):
        #Complete this implementation
        regularized_risk_val = self.risk(theta,X,Y) + lam*self.regularizer(theta)
        return regularized_risk_val
        
    def risk_grad(self,theta,X,Y): 
        #Complete this implementation
        discriminant_x = self.discriminant(theta,X) # shape: (N,1)
        exp_y_times_discriminant_x = np.exp(-Y*discriminant_x) # shape: (N,1)
        exp_common = exp_y_times_discriminant_x*(-Y)/(1+exp_y_times_discriminant_x) # shape: (N,1)
        grad_risk = exp_common*X # shape: (N,D)
        grad_risk = np.sum(grad_risk, axis=0, keepdims=True) # shape: (N,1)
        bias_val = np.sum(exp_common)
        grad_risk = np.vstack((grad_risk.T,bias_val)) # shape: (N+1,1)
        return grad_risk
        
    def regularizer_grad(self,theta):
        #Complete this implementation
        epsilon = 1e-2
        grad_regularizer = theta/np.sqrt(theta**2+epsilon)
        grad_regularizer[-1] = 0
        return grad_regularizer.reshape(-1,1) # shape: (N+1,1)

    def regularized_risk_grad(self,theta,X,Y,lam):
        #Complete this implementation
        regualrized_risk_grad_val = self.risk_grad(theta,X,Y) + lam*self.regularizer_grad(theta)
        return regualrized_risk_grad_val

    def fit(self,X,Y,lam):
        #Complete this implementation
        theta = np.zeros((X.shape[1]+1,1))
        return minimize(self.regularized_risk,theta,args=(X,Y,lam),jac=self.regularized_risk_grad,method='L-BFGS-B',tol=1e-7).x.reshape(-1,1)
        
