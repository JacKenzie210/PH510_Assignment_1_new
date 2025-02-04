#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 1- Parallel computation of Pi 
Last edited on Mon Feb 3 2025

@author: Jack MacKenzie 

Discussion and Collaboration with Ben Watson and Eamonn McHugh

"""

from mpi4py import MPI
import numpy as np
from math import fsum

#comm world communicates all mpi processes.
comm = MPI.COMM_WORLD

#number of total processors used (defined from the job script)
nproc = comm.Get_size()
print(nproc)
#number of points
N = 200000000

#Initialisation of The intergral
I = 0

#DELTA is the time step of equal length
DELTA = 1/N


#Function of the intergrand
def integrand(x):
    """
    

    Parameters
    ----------
    x : Array of points to use for the function

    Returns
    -------
    Array of Calculated estimates for each point 

    """
    return 4/(1+x**2)





#Ensures no remainders are present in x_point array so data fits into shape
N_mod = N//nproc * nproc


#points to calculate integral using the mid point rule
MID_POINT_RULE = 0.5
x_points = (np.arange(N_mod) + MID_POINT_RULE) * DELTA

#reshapes array so each row is data for a specific rank
x_points =  x_points.reshape(nproc, N_mod//nproc)

#processing the messages and work for leader
if comm.Get_rank() == 0:
    results = np.zeros(nproc)

    #send data points to workers
    for i in range(1, nproc):
        comm.send(x_points[i, :], dest=i)

    #after sending message, calculates own workload
    results[0] = fsum(integrand(x_points[0, :]) * DELTA)
    #recieve results from other processors
    for i in range(1, nproc):
        results[i] = comm.recv(source=i)

    I = fsum(results)
    print(f"Calculated Pi\n = {I:.15f} \nNumPy Pi\n = {np.pi:.15f}")
    print(f"Match = {f'{I:.15f}' == f'{np.pi:.15f}'}")
else:
    x_data = comm.recv(source = 0)
    results = fsum(integrand(x_data) * DELTA)
    comm.send(results, dest = 0)
