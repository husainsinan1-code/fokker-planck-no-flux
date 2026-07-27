import numpy as np



def constant_drift(x, value=0.0):
    return value

def linear_drift(x, slope=1.0, intercept=0.0):
    return slope * x + intercept

def constant_diffusion(x, value=1.0):
    return value

def linear_diffusion(x, slope=1.0, intercept=0.0):
    return slope * x + intercept


