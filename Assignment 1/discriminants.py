''' Import Libraries'''
import pandas as pd
import numpy as np


class Discriminant:
    ''' Prototype class for Discriminants'''
    def __init__(self):
        self.params = {}
        self.name = ''
        
    def fit(self, data):
        raise NotImplementedError
    
    def calc_discriminant(self, x):
        raise NotImplementedError



class GaussianDiscriminant(Discriminant):
    ''' Assumes a Gaussian Distribution for P(x|C_i)'''
    def __init__(self, data = None, prior=0.5, name = 'Not Defined'):
        '''Initialize pi and model parameters'''
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior}
        if data is not None:
            self.fit(data)
        self.name = name
    
    def fit(self, data):
        ''' Data is a numpy array consisting of data from a single class, where each row is a sample'''
        self.params['mu']    = np.mean(data)
        self.params['sigma'] = np.std(data)
        
        
    def calc_discriminant(self, x):
        '''Returns a discriminant value for a single sample'''
        mu = self.params['mu']
        sigma= self.params['sigma']
        prior = self.params['prior']
        '''Your code here'''
        if sigma == 0:
            sigma = 1e-12
        
        var = sigma * sigma
        term1 = -0.5 * np.log(2 * np.pi * var)
        term2 = -((x - mu) ** 2) / (2 * var)
        term3 = np.log(prior)

        g = term1 + term2 + term3

        return g
        '''Stop coding here'''


''' Create our MV Discriminant Class'''
class MultivariateGaussian(Discriminant):
    
    def __init__(self, data=None, prior=0.5, name = 'Not Defined'):
        '''Initialize pi and model parameters'''
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior}
        if data is not None:
            self.fit(data)
        self.name = name
        
    def fit(self, data):
        ''' Data is a numpy array consisting of data from a single class, where each row is a sample'''
        self.params['mu']    = np.average(data, axis=0)
        self.params['sigma'] = np.cov(data.T, bias=True)
        
    def calc_discriminant(self, x):
        mu, sigma, prior = self.params['mu'], self.params['sigma'], self.params['prior']
        '''Your code here'''
        x = np.array(x).reshape(-1)

        d = len(mu)

        det_sigma = np.linalg.det(sigma)
        if det_sigma <= 0:
            det_sigma = 1e-12

        inv_sigma = np.linalg.inv(sigma)

        diff = x - mu

        quad = diff.T @ inv_sigma @ diff

        term1 = -0.5 * np.log(det_sigma)
        term2 = -0.5 * quad
        term3 = -(d / 2) * np.log(2 * np.pi)
        term4 = np.log(prior)

        g = term1 + term2 + term3 + term4

        return g
        