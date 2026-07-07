import numpy as np



def constant_drift(x, t, value=0.0):
    return value

def linear_drift(x, t=None, slope=1.0, intercept=0.0):
    return slope * x + intercept



def constant_diffusion(x, t=None, value=1.0):
    return value

def linear_diffusion(x, t=None, slope=1.0, intercept=0.0):
    return slope * x + intercept


# PS: time is here for future implementation with time dependency 