#!/usr/bin/python

import argparse 
import os
import sys
import math
import numpy as np
from scipy import integrate
import pymbar

"""
Note the pymbar library is used in the time series analysis performed in this
script. Credit for this goes to John Chodera, Michael Shirts and Kyle Beauchamp:
    
[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples 
from multiple equilibrium states. J. Chem. Phys. 129:124105, 2008
http://dx.doi.org/10.1063/1.2978177
[2] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. 
Use of the weighted histogram analysis method for the analysis of 
simulated and parallel tempering simulations. JCTC 3(1):26-41, 2007.
"""

def parse_cmdline():
    """Command line arguments"""
    parser = argparse.ArgumentParser(description = 'Arguments')
    parser.add_argument('-d', dest = 'dir_prefix', required = True, \
                        help = 'Base directory for TI files', \
                        metavar = '<directory>')
    parser.add_argument('-l', dest = 'lbda_list', required = True, \
                        help = 'List of lambda values', metavar = '<lambdas>')
    args = parser.parse_args()
    return args  

def get_dvdl(dir_prefix, lbda):
    """Grab out dvdl values from all output files for a given lambda
    value. Removes duplicate values (assuming working with AMBER14? where
    DV/DL is printed twice at each time point (once for each of the two 
    mixed Hamiltonians"""
    
    # Initializing lists to store DV/DL with duplicates and DV/DL without them
    dvdl_forward=[]
    dvdl_forward2=[]
    
    # Loop through output files. Assuming no more than 45 of them
    # Also assumes they take the form "mdx.out" where x is the file number in 
    # sequence
    for i in range(1,45):
        if os.path.isfile(dir_prefix + "/MD/l" + str(lbda) + "/md" + \
        str(i) + ".out") == True:
            
            # Loops through and grabs every instance of DV/DL
            with open(dir_prefix + "/MD/l" + str(lbda) + "/md" + \
            str(i) + ".out", 'r') as f:
                for line in f:
                    if "DV/DL" in line:
                        dvdl_forward.append(line.split("  =      ")[-1])
            
            # Checks if simulation terminated normally. If so, removes the last 
            # 6 DV/DL values stored in dvdl_forward b/c they correspond to 
            # AVERAGES and RMS FLUCTUATIONS
            with open(dir_prefix + "/MD/l" + str(lbda) + "/md" + \
            str(i) + ".out", 'r') as f:
                for line in f:
                    if "Final Performance Info:" in line:
                        dvdl_forward = dvdl_forward[:-6]
        else:
            break

    # Remove duplicates. Store in list dvdl_forward2, which is returned
    for idx, dvdl in enumerate(dvdl_forward):
        if idx % 2 != 0:
            dvdl_forward2.append(float(dvdl))

    return dvdl_forward2

def tabulate_dvdl(dir_prefix, lbda_list):
    """Takes a list of lamda values and base directory as args.
    Iterates over all lambda points and returns 2d array of dv/dl values
    that is n x m: n is maximum number of dv/dl values present at all 
    lambda points. m represents the different lambda points considered.
    Also returns a list of lambda values"""
    
    # Obtain list of lambda values from fxn arg
    if lbda_list == "elst":
        lbda_values = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", \
                       "0.6", "0.7", "0.8", "0.9", "1.0"]
    elif lbda_list == "vdw":
        lbda_values = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", \
                       "0.6", "0.65", "0.7", "0.725", "0.75", "0.775", "0.8", \
                       "0.825", "0.85", "0.875", "0.9", "0.925", \
                       "0.95", "0.975", "1.0"]
    elif lbda_list == "rest":
        lbda_values = ["0.0", "0.01", "0.02", "0.03", "0.05", "0.075", \
                       "0.1", "0.2", "0.3", "0.4", "0.5", \
                       "0.6", "0.7", "0.8", "0.9", "1.0"]
    else:
        lbda_values = map(str, lbda_list)

    # Loop through lbda_list, perform get_dvdl, append to lbda_range
    lbda_range = []
    for i in lbda_values:
        lbda_range.append(get_dvdl(dir_prefix, i))
    
    # establish what the minimum # of dv/dl points are
    lbda_magnitude=[len(i) for i in lbda_range] 
    min_samples = min(lbda_magnitude)
    
    # Prep dv/dl data for autocorrelation analysis and descriptive stats
    # dvdl_array is 2d array with min_samples rows and len(lbda_range) cols
    dvdl_array = np.zeros((min_samples, len(lbda_range)))
    
    # populate dvdl_array
    for idx, dvdl_list in enumerate(lbda_range):
        dvdl_array[:,int(idx)] = np.asarray(dvdl_list[0:min_samples])
    
    return dvdl_array, lbda_values 

def tabulate_mean_dvdls(dvdl_array, step=500):
    """Obtain the mean dv/dl value for each simulation (all lambda points)
    correcting for the presence of correlated points at time increments of 
    "step". This is achieved by sub-sampling dv/dl values at increments of the 
    auto-correlation lagtime. The latter is computed using the pymbar package. 
    
    Returns two np.arrays that contain
    (1) The mean dv/dl values for each lambda point
    (2) The variance of dv/dl values for each lambda
    Both arrays are 2-D, where rows are means computed at increments of "step"
    and cols are lambda points"""
    
    #dvdl_array = tabulate_dvdl("vdw")[0]
    
    # List of time intervals over which <dv/dl> will be computed
    stepped_values = []
    for i in range(len(dvdl_array[:,0])/step):
        stepped_values.append((i+1)*step-1)

    # initializing uncorr_mean & uncorr_var arrays to hold the mean/variance 
    # at intervals of "step", removing correlated points
    # taus is initialized to hold the autocorrelation lagtimes, for referencing
    # when removing those correlated points
    a = len(dvdl_array[:,0])/step
    b = len(dvdl_array[0,:])
    
    uncorr_mean = np.zeros([(len(dvdl_array[:,0])/step), len(dvdl_array[0,:])])
    uncorr_var = np.zeros([(len(dvdl_array[:,0])/step), len(dvdl_array[0,:])])
    taus = np.zeros(len(dvdl_array[0,:]))
    
    # Loop through lambdas (columns of dv/dl array) and compute autocorr lag,
    # store it in taus array, then loop through dv/dl values,computing the mean
    # for samples of length increment step (every 0.5 ns if step = 500,removing
    # correlated points (sampling at increments of tau)
    
    for i in range(len(dvdl_array[0,:])):
        taus[i] = \
        np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dvdl_array[:,i]))
    
        for j in range(len(uncorr_mean[:,0])):
            uncorr_mean[j,i] = np.mean(dvdl_array[:stepped_values[j]:taus[i],i])
            uncorr_var[j,i] = np.var(dvdl_array[:stepped_values[j]:taus[i],i])        
    
    return uncorr_mean, uncorr_var, taus

def variance_weights(lbdas):
    """Takes a list of lambda points and returns the weights to be 
    applied in the weighted sum for computing the variance of the 
    TI transformation under study."""
    
    # initialize array to hold weights
    weights = np.zeros(len(lbdas))
    
    # compute statistical weights. since using trapezoidal rule, this is 
    # equivalent to ((lambda2 - lambda1)/2)^2 for endpoint lambdas 
    # (l =0 and l = 1). Equal to (lambda2-lambda1)^2 for non-endpoint lambdas
    
    weights[0] = (0.5*(lbdas[1] - lbdas[0]))**2
    for i in range(1, len(lbdas) - 1):
        weights[i] = (lbdas[i] - lbdas[i-1])**2
    weights[-1] = (0.5*(lbdas[-1] - lbdas[-2]))**2
    
    return weights

def tabulate_free_energies(dir_prefix, lbda_list):
    
    # Create dv/dl array and get lambdas
    dvdl_array, lbdas = tabulate_dvdl(dir_prefix, lbda_list)
    lbdas = np.array(map(float, lbdas))
    
    # get statistical weights for lambdas
    weights = variance_weights(lbdas)
    
    # get mean and variance arrays
    uncorr_mean, uncorr_var, taus = tabulate_mean_dvdls(dvdl_array)
    
    # initialize array to hold dG values and standard deviations
    dG = np.zeros((len(uncorr_mean[:,0]),3))
    
    # loop through mean and variance arrays. Compute dG by integrating over 
    # all <dv/dl> values Get standard deviation by multiplying variance of 
    # each simulation by its statistical weight.
    # Performing this at increments of "step". By default, this is 500 
    # - which is every 0.5 ns - given how frequently you write to md.out
    
    for i in range(len(uncorr_mean[:,0])):
        # write out time (increments of 0.5 ns)
        dG[i,0] = (i+1)*0.5
        # get dG
        integrated = integrate.cumtrapz(uncorr_mean[i,:],lbdas[:])
        dG[i,1] = integrated[-1]
        # get its sd
        dG[i,2] = np.sqrt(sum(weights*uncorr_var[i,]))
                              
    return dG

if __name__ == "__main__":
    args = parse_cmdline()
    dir_prefix = args.dir_prefix
    lbda_list = args.lbda_list
    x = tabulate_free_energies(dir_prefix, lbda_list)
    print "{} \t {} \t\t {}".format('Time(ns)', 'dG', 'sd')
    for c1, c2, c3 in x:
        print "{} \t\t {} \t {}".format(c1, round(c2, 4), round(c3, 4))

