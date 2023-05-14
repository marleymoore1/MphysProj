#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:33:31 2022

@author: marleymoore
"""
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from matplotlib import pyplot as plt, patches
#%%
# Control Panel

QUANTITY = 500 #How many divisions of the total sedimentation
HEAD = 12 #Display the first how many
fibre_type = 'flat' #'flat' or 'angled'

#Make sure tje following parameters match with those used when running simuations
REPULSION_STRENGTH = 1
STIFFNESS_PARAMETER = 2
CONTACT_DISTANCE_FACTOR =1.1
ETA = 1
WEIGHT = 1
BE = 1000
RADIUS = 1e-3
CENTRELINE_DISTANCE_FACTOR = 2

single_fibre_bead_number = 20

#If plotting how Be affects shape, input the range of Be used
RANGE_OF_BE = [-3, -2, -1, 0, 1, 2, 3, 4]

plot_2D = False
plot_3D = False
plot_angle_development = False
plot_log_angles = False
plot_symmetry_evolution = False
plot_change_of_length = True

plot_fibre_length_affects_min_angle = False
plot_fibre_length_affects_velocity = False
plot_Be_range = False
#%%
# os.chdir('/Users/marleymoore/Desktop/University of Manchester/Year 4/MPhys Project/Code/BeadSpringModex/multi_fibers/output')
# df1 = pd.read_csv('run.output_positions.csv', delimiter=';', header=None)
# df2 = pd.read_csv('run.output_times.csv', delimiter=';', header=None)
# fibre_pos = df1
# fibre_tim = df2

def read_csv(number_of_beads):
    if fibre_type == 'flat':
        os.chdir('/Users/marleymoore/Desktop/University of Manchester/Year 4/MPhys Project/Code/BeadSpringModex/multi_fibers/output/flat_fibres')
        pos_df = pd.read_csv('Flat_{}_bead_positions.csv'.format(number_of_beads),
                             delimiter=';', header=None)
        tim_df = pd.read_csv('Flat_{}_bead_times.csv'.format(number_of_beads),
                         delimiter=';', header=None)
    
    elif fibre_type == 'angled':
        os.chdir('/Users/marleymoore/Desktop/University of Manchester/Year 4/MPhys Project/Code/BeadSpringModex/multi_fibers/output/angled_fibres')
        pos_df = pd.read_csv('Angled_{}_bead_positions.csv'.format(number_of_beads),
                             delimiter=';', header=None)
        tim_df = pd.read_csv('Angled_{}_bead_times.csv'.format(number_of_beads),
                         delimiter=';', header=None)
        
    #Removing last row which contains only zeros
    pos_df = pos_df.iloc[:-1, :]
    tim_df = tim_df.iloc[:-1, :]
    return pos_df, tim_df

def read_Be_csv(Be):
    os.chdir('/Users/marleymoore/Desktop/University of Manchester/Year 4/MPhys Project/Code/BeadSpringModex/multi_fibers/output/Changing_Be')
    pos_df = pd.read_csv('1e{}Be_positions.csv'.format(Be),
                         delimiter=';', header=None)
    tim_df = pd.read_csv('1e{}Be_times.csv'.format(Be),
                         delimiter=';', header=None)
    pos_df = pos_df.iloc[:-1, :]
    tim_df = tim_df.iloc[:-1, :]
    return pos_df, tim_df

#Reading in the csv files that contain the position and time data as pandas dataframes
fibre_pos, fibre_tim = read_csv(single_fibre_bead_number)

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

def dimension_values(dimension, position_df):
    # Picks out the position values of the given dimension for all times
    idx_array = every_three(len(position_df.iloc[0,:]), dimension)
    values = position_df.iloc[:, idx_array]
    
    return values

def plot_3D_positions(quantity, position_df):
    # Plots the fiber in 3D at a number of times given by 'quantity'
    fig2 = plt.figure()
    ax1 = Axes3D(fig2)
    
    x_values = dimension_values('x', position_df)
    y_values = dimension_values('y', position_df)
    z_values = dimension_values('z', position_df)
    
    for i in range(0,len(x_values), int((len(x_values))/quantity)):
        ax1.plot(x_values.iloc[i,:], y_values.iloc[i,:], z_values.iloc[i,:])

    # ax1.set_xticks(np.arange(-1,1,0.05))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    plt.show()

#%%
def plot_2D_positions(quantity, head, position_df):
    # Plots the fiber in 2D at a number of times given by 'quantity'
    x_values = dimension_values('x', position_df)
    z_values = dimension_values('z', position_df)
    
    fig1, axs = plt.subplots(1,2, figsize=[8,6])
    
    for i in range(0,len(x_values), int((len(x_values))/quantity)):
        axs[0].plot(x_values.iloc[i,:],z_values.iloc[i,:])
    
    for i in range(0, head*int((len(x_values))/quantity), int((len(x_values))/quantity)):
        axs[1].plot(x_values.iloc[i,:],z_values.iloc[i,:])
        for j in range(len(x_values.iloc[i,:])):
            circle = patches.Circle((x_values.iloc[i,j], z_values.iloc[i,j]),
                                    radius=RADIUS, color='red', fill=False)
            axs[1].add_patch(circle)
    
    # plt.xticks(range(x_min, x_frames.iloc[0,:][-1], max(x_frames.iloc[0,:])/20))
    axs[0].set_xlabel('X / m')
    axs[0].set_ylabel('Z / m')
    axs[0].set_title('Filament as it sediments in time')
    # axs[0].axis('equal')
    
    axs[1].set_xlabel('X / m')
    # axs[1].set_ylabel('Z / m')
    axs[1].set_title('First {} positions'.format(head))
    axs[1].axis('equal')
    
    plt.show()

#%%
def endpoints(position_df):
    # Finds the positions in 2D of the ends of the fibre at all times
    x_values = dimension_values('x', position_df)
    z_values = dimension_values('z', position_df)
    
    z_beginnings = z_values.iloc[:,0]
    z_ends = z_values.iloc[:,-1]
    
    x_beginnings = x_values.iloc[:,0]
    x_ends = x_values.iloc[:,-1]
    
    return x_beginnings, x_ends, z_beginnings, z_ends
    
def lowest_points(position_df):
    # Finds the positions in 2D of the lowest point of the fibre at all times
    z_values = dimension_values('z', position_df)
    x_values = dimension_values('x', position_df)
    
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
    # other dimensions, need to take 2 away from each z idx number
    x_min_idx_list = [z-2 for z in z_min_idx_list]
    x_mins = []
    
    # Finding the value of x at each idx that corresponds to the minimum z
    for i, idx in enumerate(x_min_idx_list):
        x_min = x_values.loc[i, idx]
        x_mins.append(x_min)
        
    return z_mins, x_mins

def pythagoras(x_start, x_end, z_start, z_end):
    x_difference = x_end - x_start
    z_difference = z_end - z_start
    
    side_length = np.sqrt(x_difference**2 + z_difference**2)
    return side_length

def angle(c, b, a):
    # Cosine rule
    theta = np.arccos((c**2 - b**2 - a**2)/(-2*b*a))
    return theta
    
def angle_between(position_df):
    # Fetching the positions of the ends and the beginnings and the middles of the fibre
    # as it falls
    x_beginnings, x_ends, z_beginnings, z_ends = endpoints(position_df)
    z_mins, x_mins = lowest_points(position_df)

    angle_list = []
    for i in range(len(x_beginnings)):
        MA = pythagoras(x_beginnings[i], x_mins[i], z_beginnings[i], z_mins[i])
        MB = pythagoras(x_ends[i], x_mins[i], z_ends[i], z_mins[i])
        AB = pythagoras(x_ends[i], x_beginnings[i], z_ends[i], z_beginnings[i])
        theta = angle(AB, MB, MA)
        theta = math.degrees(theta)
        angle_list.append(theta)
    
    return angle_list

def plot_angle(quantity, head, position_df, time_df, show_plot=False):
    # Plots the angle between beginning, low, and ends points to see how it evolves
    # in time
    x_beginnings, x_ends, z_beginnings, z_ends = endpoints(position_df)
    z_mins, x_mins = lowest_points(position_df)
    
    x_values = dimension_values('x', position_df)
    z_values = dimension_values('z', position_df)
    
    # plotting first few fibre positions and "v" shaped angle
    fig3, axs = plt.subplots(1,2, figsize=[8,6])
    
    for i in range(0, head*int((len(x_values))/quantity), int((len(x_values))/quantity)):
        x_beginning = x_beginnings[i]
        x_minimum = x_mins[i]
        x_end = x_ends[i]
        x_pos = np.array([x_beginning, x_minimum, x_end])  
        
        z_beginning = z_beginnings[i]
        z_minimum = z_mins[i]
        z_end = z_ends[i]
        z_pos = np.array([z_beginning, z_minimum, z_end]) 
        
        if show_plot==True:
            axs[0].plot(x_pos, z_pos, 'r')
            axs[0].plot(x_values.iloc[i,:],z_values.iloc[i,:])     
    
    # calculating the angle
    angle_list = angle_between(position_df)
    angle_list = angle_list[:head*int((len(x_values))/quantity)]
    time_list = time_df.iloc[:,0][:head*int((len(x_values))/quantity)]
        
    
    # ridding of any nan values
    angles = []
    for angle in angle_list:
        if str(angle) != 'nan':
            angles.append(angle)
    min_angle = min(angles)
    
    # plotting angle evolution
    if show_plot==True:
        axs[1].plot(time_list, angle_list, 'r')
        axs[1].set_xlabel('Time / s')
        axs[1].set_ylabel('"V" shaped angle / degrees')
        # axs[1].text(0.1, 150, 'Minimum angle: {:,.1f} degrees'.format(min_angle))
    
        axs[0].set_xlabel('X / m')
        axs[0].set_ylabel('Z / m')
    
        axs[0].set_title('The "V" shape')
        axs[1].set_title('Evolution of "V" shaped angle')
        plt.show()
        print('Minimum angle: {:,.1f} degrees'.format(min_angle))
    
    return min_angle

def plot_log_angle(quantity, head, position_df, time_df):
    # Plotting the log of the angles over time
    angle_list = angle_between(position_df)
    angle_list = angle_list[:head*int((len(time_df))/quantity)]
    
    log_angle_list = np.log(angle_list)
    time_list = time_df.iloc[:,0][:head*int((len(time_df))/quantity)]
    
    plt.plot(time_list, log_angle_list, 'r')
    plt.xlabel('Time / s')
    plt.ylabel('ln(angle)')
    plt.title('Log angle evolution')
    
    plt.grid(b=True, which='major', color='gray', alpha=0.6,
             linestyle='dashdot', lw=1.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8,
             ls='-', lw=1)
    plt.show()
#%%
# Symmetry evolution
def plot_symmetry(head, position_df, time_df):
    #sees how symmetry evolves in time
    x_beginnings, x_ends, z_beginnings, z_ends = endpoints(position_df)
    z_mins, x_mins = lowest_points(position_df)
    
    assert len(position_df)==len(time_df) 
    time_list = time_df.iloc[:,0]
    
    ratio_list = []
    for i in range(len(position_df)):
        MA = pythagoras(x_beginnings[i], x_mins[i], z_beginnings[i], z_mins[i])
        MB = pythagoras(x_ends[i], x_mins[i], z_ends[i], z_mins[i])
        ratio = MB/MA
        ratio_list.append(ratio)
    
    plt.plot(time_list[:head], ratio_list[:head], 'r')
    plt.plot(time_list[:head], np.ones_like(ratio_list[:head]), 'k', ls='--')
    plt.xlabel('Time / s')
    plt.yticks(np.arange(0,2.1,0.2))
    plt.ylabel('Ratio of sides')
    plt.title('Symmetry evolution')
    plt.show()

#%%
# Overall length evolution
def plot_length_change(head, position_df, time_df):
    z_values = dimension_values('z', position_df)
    x_values = dimension_values('x', position_df)
    
    assert len(position_df)==len(time_df) 
    time_list = time_df.iloc[:,0]
    
    fibre_lengths = []
    for fibre in range(len(x_values.iloc[:,0])):
        bead_separations = []
        
        for j in range(len(x_values.iloc[0,:])-1):
            bead_separation = pythagoras(x_values.iloc[fibre, :].iloc[j],
                                         x_values.iloc[fibre, :].iloc[j+1],
                                         z_values.iloc[fibre, :].iloc[j],
                                         z_values.iloc[fibre, :].iloc[j+1])
            bead_separations.append(bead_separation)
        
        fibre_length = np.sum(bead_separations)
        fibre_lengths.append(fibre_length)
    
    ones = np.ones_like(fibre_lengths[:head])
    raw_length = ones * single_fibre_bead_number * RADIUS * 2
    
    plt.plot(time_list.iloc[:head], raw_length, 'k', ls='--', label = 'raw length')
    plt.plot(time_list.iloc[:head], fibre_lengths[:head], 'r', label = 'summed length')
    plt.xlabel('Time / s')
    plt.ylabel('Fibre length / m')
    plt.yticks(np.arange(0.03, 0.055, 0.005))
    plt.legend()
    plt.title('Length evolution')
    print('Maximum length = {:.3g} cm'.format(max(fibre_lengths)*100))
    print('Minimum length = {:.3g} cm'.format(min(fibre_lengths)*100))
    print('Fibre contracts by {:.2g}%'.format(100*(max(fibre_lengths)-min(fibre_lengths))/max(fibre_lengths)))
    plt.show()

#%%
# def polyfit_to_fibres(degrees, position_df):
#     #selecting relavent parts of the dataframe 
#     x_values = dimension_values('x')
#     z_values = dimension_values('z')
    
#     #using the polyfit function to obtain the coefficients of a polynomial that best fits each fibre
#     coefficients = []
#     for i in range(len(x_values)):
#         p = np.poly1d(np.polyfit(x_values.iloc[i,:], z_values.iloc[i,:], deg=degrees))
#         coefficients.append(p)
    
#     # Coefficients in the form Ax^3 + Bx^2 + Cx + D
#     if degrees == 3:
#         As = []
#         Bs = []
#         Cs = []
#         Ds = []
#         for i in range(len(coefficients)):
#             row = coefficients[i]
#             A = row[degrees]
#             B = row[degrees-1]
#             C = row[degrees-2]
#             D = row[degrees-3]
#             As.append(A)
#             Bs.append(B)
#             Cs.append(C)
#             Ds.append(D)
        
#         return As, Bs, Cs, Ds
    
#     elif degrees == 2:
#         Bs = []
#         Cs = []
#         Ds = []
#         for i in range(len(coefficients)):
#             row = coefficients[i]
#             B = row[degrees]
#             C = row[degrees-1]
#             D = row[degrees-2]
#             Bs.append(B)
#             Cs.append(C)
#             Ds.append(D)
        
#         return Bs, Cs, Ds
            
#     elif degrees == 1:
#         Cs = []
#         Ds = []
#         for i in range(len(coefficients)):
#             row = coefficients[i]
#             C = row[degrees]
#             D = row[degrees-1]
#             Cs.append(C)
#             Ds.append(D)
        
#         return Cs, Ds

# def plot_polynomials(quantity, head):
#     # Coefficients in the form Ax^3 + Bx^2 + Cx + D
#     A_cubic, B_cubic, C_cubic, D_cubic = polyfit_to_fibres(3)
#     B_sqare, C_square, D_square = polyfit_to_fibres(2)
#     C_linear, D_linear = polyfit_to_fibres(1)
    
#     x_values = dimension_values('x')
#     z_values = dimension_values('z')
    
#     fig1, axs = plt.subplots(1,3, figsize=[8,6])
    
#     # for each of the first few fibre positions, this plots the best fit polynomial against the fibre positions to 
#     # show the accuracy of each fit
#     for i in range(0, head*int((len(df2))/quantity), int((len(df2))/quantity)):
#         cubic_polynomial = A_cubic[i]*(x_values.iloc[i,:])**3
#         + B_cubic[i]*(x_values.iloc[i,:])**2 + C_cubic[i]*(x_values.iloc[i,:])
#         + D_cubic[i]
        
#         quadratic_polynomial = B_sqare[i]*(x_values.iloc[i,:])**2
#         + C_square[i]*(x_values.iloc[i,:]) + D_square[i]
        
#         linear_polynomial = C_linear[i]*(x_values.iloc[i,:]) + D_linear[i]

#         axs[0].plot(x_values.iloc[i,:],z_values.iloc[i,:])
#         axs[0].plot(x_values.iloc[i,:],cubic_polynomial, 'r')
        
#         axs[1].plot(x_values.iloc[i,:],z_values.iloc[i,:])
#         axs[1].plot(x_values.iloc[i,:],quadratic_polynomial, 'r')
        
#         axs[2].plot(x_values.iloc[i,:],z_values.iloc[i,:])
#         axs[2].plot(x_values.iloc[i,:],linear_polynomial, 'r')
    
#     axs[0].set_ylabel('Z / m')
#     axs[0].set_xlabel('X / m')
#     axs[1].set_xlabel('X / m')
#     axs[2].set_xlabel('X / m')
    
#     axs[0].set_title('Fitting cubic')
#     axs[1].set_title('Fitting quadratic')
#     axs[2].set_title('Fitting linear')

#     plt.show()
# plot_polynomials(quantity=50, head=5)
#%%
#How fibre length affects min angle and velocity
def odd_numbers(start, end):
    odd_numbers = []
    for num in range(start, end+1):
        if num % 2 != 0:
            odd_numbers.append(num)
    return odd_numbers

def minimum_angles(start, end):
    # odd_fibre_lengths = odd_numbers(start, end)
    fibre_lengths = np.arange(start, end)
    min_angles = []
    for length in fibre_lengths:
        positions, times = read_csv(length)
        min_angle = plot_angle(QUANTITY, HEAD, positions, times, show_plot=False)
        min_angles.append(min_angle)
    
    return min_angles, fibre_lengths

def plot_min_angles(start, end):
    min_angles, fibre_lengths = minimum_angles(start, end)
    plt.figure()
    plt.plot(fibre_lengths, min_angles, 'r')
    plt.xlabel('Number of beads in fibre')
    plt.ylabel('Minimum angle reached / degrees')
    plt.title('Equilibrium angle changes with fibre length')
    plt.show()
    return None

def plot_velocity(start, end):
    fibre_lengths = np.arange(start, end)
    velocities = []
    for length in fibre_lengths:
        positions, times = read_csv(length)
        z_mins, x_mins = lowest_points(positions)
        minimum_z_min = z_mins[-1]
        maximum_z_min = z_mins[0]
        
        velocity = (np.abs(minimum_z_min-maximum_z_min))/times.iloc[-1]
        velocities.append(velocity)
        
    plt.figure()
    plt.plot(fibre_lengths, velocities, 'r')
    plt.xlabel('Number of beads in fibre')
    plt.ylabel('Sedimentation velocity / m/s')
    plt.title('Sedimentation velocity changes with fibre length')
    plt.show()
#%%
#Changing the elasto-gravitational number affects the shape
def slender_body_theory(x):
    A = x**2
    B = x+1
    C=2*np.log(2)
    
    y = (1/24) * (A + ((13/6)*A**2)) + C*(6*A+A**2) - ((x-1)**4) 
    return y

def plot_final_pos_different_Be(): 
    fig10, axs = plt.subplots(1,1, figsize=[8,6])
    color_list = ['r', 'b', 'k', 'brown',
                  'm', 'c', 'y', 'g']
    for Be in RANGE_OF_BE:
        positions, times = read_Be_csv(Be)
        
        x_values = dimension_values('x', positions)
        z_values = dimension_values('z', positions)
        final_xs = x_values.iloc[-1,:]
        final_zs = z_values.iloc[-1,:]
        
        z_mins, x_mins = lowest_points(positions)
        final_x_min = x_mins[-1]
        final_z_min = z_mins[-1]
        
        rescaled_final_xs = final_xs - final_x_min
        rescaled_final_zs = final_zs - final_z_min
        
        axs.plot(rescaled_final_xs, rescaled_final_zs, color=color_list[Be],
                 label='Be = 1e{}'.format(Be))
        for j in range(len(x_values.iloc[-1,:])):
            circle = patches.Circle((rescaled_final_xs.iloc[j], rescaled_final_zs.iloc[j]),
                                radius=RADIUS, color=color_list[Be], fill=False)
            axs.add_patch(circle)
    
    slender_body_x_pos = 1
    slender_body_z_pos = slender_body_theory(slender_body_x_pos)
    axs.plot()  
    
    axs.set_xlabel('X / m')
    axs.set_ylabel('Z / m')
    axs.set_title('Final position at a range of Be')
    axs.axis('equal')
    axs.legend()
    plt.show()
#%%
if plot_2D == True:
    plot_2D_positions(quantity=QUANTITY, head=HEAD, position_df=fibre_pos)

if plot_3D == True:
    plot_3D_positions(quantity=QUANTITY, position_df=fibre_pos)
    
if plot_angle_development == True:
    plot_angle(quantity=QUANTITY, head=HEAD, position_df=fibre_pos, time_df=fibre_tim, show_plot=True)

if plot_log_angles == True:
    plot_log_angle(QUANTITY, HEAD, position_df=fibre_pos, time_df=fibre_tim)
    
if plot_symmetry_evolution == True:
    plot_symmetry(2900, position_df=fibre_pos, time_df=fibre_tim)  
    
if plot_change_of_length == True:
    plot_length_change(head=10, position_df=fibre_pos, time_df=fibre_tim)
    
if plot_fibre_length_affects_min_angle == True:
    plot_min_angles(5, 20)
    
if plot_fibre_length_affects_velocity == True:
    plot_velocity(5, 20)

if plot_Be_range == True:
    plot_final_pos_different_Be()
#%%






        
        
