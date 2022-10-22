#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:33:31 2022

@author: marleymoore
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#%%
os.chdir('/Users/marleymoore/Desktop/University of Manchester/Year 4/MPhys Project/Code/BeadSpringModex/multi_fibers/output')

df1 = pd.read_csv('run.output_positions.csv', delimiter=';', header=None)
df2 = pd.read_csv('run.output_times.csv', delimiter=';', header=None)

#Removing last row which contains only zeros
df1 = df1.iloc[:-1, :]
df2 = df2.iloc[:-1, :]
#%%
def every_three(length, dimension):
    # Produces arrays of indices so we can select out particular columns of dataframe
    if dimension == 'x':
        idx = np.arange(0, int(length/3), 1)
        idx_array = 3*idx
        
    elif dimension == 'y':
        idx_array = np.zeros(int(length/3))
        idx_array[0]=1
        for i in range(len(idx_array)-1):
            idx_array[i+1] = idx_array[i]+3
    
    elif dimension == 'z':
        idx_array = np.zeros(int(length/3))
        idx_array[0]=2
        for i in range(len(idx_array)-1):
            idx_array[i+1] = idx_array[i]+3

    return idx_array

def dimension_values(dimension):
    # Picks out the position values of the given dimension for all times
    idx_array = every_three(len(df1.iloc[0,:]), dimension)
    values = df1.iloc[:, idx_array]
    
    return values
    
def plot_2D_positions(quantity):
    # Plots the fiber in 2D at a number of times given by 'quantity'
    x_values = dimension_values('x')
    z_values = dimension_values('z')
    
    for i in range(0,len(df2), int((len(df2))/quantity)):
        plt.plot(x_values.iloc[i,:],z_values.iloc[i,:])
    
    # plt.xticks(range(x_min, x_frames.iloc[0,:][-1], max(x_frames.iloc[0,:])/20))
    plt.xlabel('X / m')
    plt.ylabel('Z / m')
    plt.title('Filament as it sediments in time')
    
    plt.show()

def plot_3D_positions(quantity):
    # Plots the fiber in 3D at a number of times given by 'quantity'
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    
    x_values = dimension_values('x')
    y_values = dimension_values('y')
    z_values = dimension_values('z')
    
    for i in range(0,len(df2), int((len(df2))/quantity)):
        ax1.plot(x_values.iloc[i,:], z_values.iloc[i,:], y_values.iloc[i,:])

    # ax1.set_xticks(np.arange(-1,1,0.05))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    
    plt.show()

def endpoints():
    # Finds the positions in 2D of the ends of the fibre at all times
    x_values = dimension_values('x')
    z_values = dimension_values('z')
    
    z_beginnings = z_values.iloc[:,0]
    z_ends = z_values.iloc[:,-1]
    
    x_beginnings = x_values.iloc[:,0]
    x_ends = x_values.iloc[:,-1]
    
    return x_beginnings, x_ends, z_beginnings, z_ends
    
def lowest_points():
    # Finds the positions in 2D of the lowest point of the fibre at all times
    z_values = dimension_values('z')
    x_values = dimension_values('x')
    
    # Calculating minimums of each row and storing as a list
    minimum_z_values = z_values.min(axis=1)
    z_mins = list(minimum_z_values)
    
    
    z_min_idx_list = []
    # Finding the column name of the minimum z_value for each row
    for row in range(len(minimum_z_values)):
        mask = z_values.iloc[row,:] == minimum_z_values.iloc[row]
        min_idx = np.where(mask)[0][0]
        col_name_of_min = z_values.columns.tolist()[min_idx]
        z_min_idx_list.append(col_name_of_min) 
    
    # Each index in z corresponds to correct column after dropping columns of 
    # other dimensions
    x_min_idx_list = [z-2 for z in z_min_idx_list]
    
    x_mins = []
    
    # Finding the value of x at each idx that corresponds to the minimum z
    for i, idx in enumerate(x_min_idx_list):
        x_min = x_values.loc[i, idx]
        x_mins.append(x_min)
        
    return z_mins, x_mins

def angle_between():
    x_beginnings, x_ends, z_beginnings, z_ends = endpoints()
    z_mins, x_mins = lowest_points()
    
    AB_list = []
    for i in range(len(df1)):
        AB = np.sqrt((x_beginnings[i]-x_mins[i])**2+()**2)
    
    

        
    
    
    


#%%    
plot_2D_positions(len(df1.iloc[0,:]), quantity=50)
# plot_3D_positions(len(df1.iloc[0,:]), quantity=50)