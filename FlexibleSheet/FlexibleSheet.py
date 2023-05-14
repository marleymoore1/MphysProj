#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:31:49 2023

@author: Marley Moore

READ ME

This script produces an array of 3D position arrays that describe how the shape
of an elastic sheet in a viscous fluid evolves over time when subjected to
gravity. 

The variables of the simulation can be adjusted in the 'globals'
tab. However, it should be noted that at small lattice sizes, sheets are prone
to buckling and crumpling, and results appear to blow up rapidly. Larger lattices
represent more realistic sheets, but take longer to simulate. Also true is that 
at large timesteps, the simulation becomes unrealistic. A small time step must 
be used, in order to apply the elastic forces rapidly, before they have time to
blow up in size due to overcorrection. This is explained in more detail in the
accompanying report.

The shape progression should then be plotted in your favourite way, but I have
included a plotting script in the GitHub if that is easier. Because the output
positions are returned via Pickle dump, it is important to use Pickle to load
the positions into a plotting script, i.e. using:

with open(output_path, "rb") as f:
    list_of_positions = pickle.load(f)

Also attached is an accompanying report that explains the simulation process
and the drawbacks of this method and/or possible issues.

Notes:
    This script requires the helper script FlexibleSheet_helper to provide the
    bending force function.
    
    This simulation only includes the RPY tensor for non-overlapping beads.
    This can easily be adjusted to account for cases in which the bead
    separation is allowed to be less than twice the bead radius. See
    
    The effects of buoyancy are ignored in this simulation.
    
    The effects of thermal fluctuations are not included in this model.    
    
"""
#%% IMPORTS
import numpy as np
import pandas as pd
import math
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from FlexibleSheet_helper import unit_cell_bending_force

#%% GLOBAL PARAMETERS
M = 1e-3 #sheet mass (kg)
R = 1e-3 #bead radius (m)
r0 = 2*R #rest length of springs (minimum 2R)
MU = 1 #dynamic viscosity of fluid (kg m/s)
E = 0 #3D Young's modulus of material (N/m^2)
NU = 0.3 #Poisson ratio 
h = 1e-7 #sheet thickness (m)

TILT = 0.3 #angle to horizontal in radians (rotation about y-axis)

T=0.1 #total sedimentation time (s)
H=0.0001 #timestep for numerical integration (s) (use ~ 10^-4)

# where would you like to save the position data?
SAVE_DIR = '/Users/marleymoore/Desktop/University of Manchester/Year 4/MPhys Project/Semester 2'

SHEET_TYPE = 'hexagon' #'hexagon' or 'fibre'
if SHEET_TYPE == 'hexagon':
    EDGE_LENGTH = 15 #number of beads on the hexagon's edge (>1)
    assert EDGE_LENGTH > 1
elif SHEET_TYPE == 'fibre': #sediment a long, thin fibre
    FIBRE_LENGTH = 50 #number of beads along the fibre's long edge
    FIBRE_WIDTH = 3 #number of beads along the width. Should be odd, and > 2
    assert FIBRE_WIDTH > 2
else:
    raise IOError('Only hexagon or fibre options available')

# scatter plot of initial bead positions
DIAGRAM_PATH = Path(SAVE_DIR) / 'graphs' / 'initial_lattice_diagram.png'
DIAGRAM_PATH.parent.mkdir(exist_ok=True, parents=True)
 
# saves all the positions as a text file via pickle dump 
OUTPUT_PATH = Path(SAVE_DIR) / 'runs' / 'output_positions.txt' 
OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)

#%% ESTABLISH LATTICE
def initial_hexagon_positions(edge_length=12):
    """
    This function produces an array of positions (each a 3-vector) of all the
    beads in the sheet. It also produces the positions of the beads split into
    the horizontal 'rows' of the hexagon to which the beads belong.
    
    The origin is placed at the centre of the hexagon in the x-dimension, and
    at the edge in the y-dimension.
    
    An edge length of 30 returns a hexagonal lattice of beads which has a 
    circumradius of 59*A (i.e. the hexagon is 59 beads wide at its largest,
    30 at its smallest). 
    """
    rows = []
    
    # setting up empty arrays for top half of hexagon
    for i in range(edge_length):
        row_i = np.zeros((edge_length+i,3))
        rows.append(row_i)
    
    # setting up arrays for bottom half of hexagon
    for i in range(edge_length-2,-1,-1):
        row_mid_plus_i = np.zeros((edge_length+i,3))
        rows.append(row_mid_plus_i)
    
    # filling in top half with position values (including middle row)
    for i in range(edge_length):
        # filling x-coordinates
        rows[i][:,0] = r0 * np.arange(-((edge_length-1)/2)-(i*0.5),
                                      ((edge_length+1)/2)+(i*0.5))
        # filling y-coordinates
        rows[i][:,1] = r0 * i*np.sqrt(3)/2
    
    # filling in bottom half with position values
    for i in range(edge_length-1):
        # filling x-coordinates
        rows[edge_length+i][:,0] = r0 * np.arange(-(edge_length-1.5)+(i*0.5),
                                                  (edge_length-0.5)-(i*0.5))
        # filling y-coordinates
        rows[edge_length+i][:,1] = r0 * (edge_length+i)*np.sqrt(3)/2
    
    # collapsing all rows into one array of positions
    all_beads = [item for sublist in rows for item in sublist]
        
    return rows, all_beads

def initial_fibre_positions(fibre_length=50, fibre_width=3):
    """
    This function produces an array of positions (each a 3-vector) of all the 
    beads that form a thin fibre. The fibre length should be >> fibre width in 
    order to reproduce the results of fibre sedimentation. 
    
    This function only produces fibres of odd width, so forces even input into
    odd.
    """
    rows = []
    
    # force even width input into odd
    if fibre_width % 2 == 0:
        fibre_width += 1
    
    fibre_width = int(fibre_width)    
    
    # number of layers up to middle
    half_minus = int((fibre_width-1)/2)
    
    # setting up the top half of the fibre
    for i in range(half_minus, -1, -1):
        row_i = np.zeros((fibre_length - i, 3))
        rows.append(row_i)
    
    # setting up the top half of the fibre
    for i in range(1, half_minus+1):
        row_mid_plus_i = np.zeros((fibre_length - i, 3))
        rows.append(row_mid_plus_i)
    
    # filling in middle row
    rows[half_minus][:,0] = r0 * np.arange(fibre_length)
    
    # filling in other rows
    for i in range(1,half_minus+1):
        #filling in x-coordinates
        rows[half_minus - i][:,0] = r0 * np.arange(0.5*i,
                                                   fibre_length - (0.5*i))
        rows[half_minus + i][:,0] = r0 * np.arange(0.5*i,
                                                   fibre_length - (0.5*i))
        
        #filling in y-coordinates
        rows[half_minus - i][:,1] = r0 * i * np.sqrt(3)/2 
        rows[half_minus + i][:,1] = -r0 * i * np.sqrt(3)/2 

    # collapsing all rows into one array of positions
    all_beads = [item for sublist in rows for item in sublist]
        
    return rows, all_beads

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def plot_initial_lattice(positions):
    """
    Plot the beads as a scatter plot.
    """
    plt.close('all')
    ax, fig = plt.subplots(figsize=(12,12))
    positions = pd.DataFrame(positions, columns=['x','y','z'])
    sns.scatterplot(data=positions, x='x',y='y', color='blue', 
                    alpha=0.5, s=600)
    
    # plotting the index of the bead on top of the scatter plot
    for i in positions.index:
        plt.text(positions.loc[i,'x'],
                 positions.loc[i,'y'],
                 str(i), ha='center', va='center', fontsize='20')
        
    plt.axis('equal')
    plt.xlabel('x', fontsize=30)
    plt.ylabel('y', fontsize=30)
    
    # save the figure as a png    
    plt.savefig(DIAGRAM_PATH, format='png', dpi=600)
    
    return None

def perimeter_separation(rows, all_beads):
    """
    This function returns the indices of the perimeter beads in an arbitrary
    triangular lattice, given the rows and the list of beads.
    
    The corner beads only have three springs whereas generic perimeter beads
    will have four, so this function separates them from one another too.
    """
    perimeter_bead_index_list = []
    
    # first row
    for i in range(len(rows[0])-1):
        perimeter_bead_index_list.append(i)
        
    # first and last bead in each middle row
    for i in range(1, len(rows)):
        row_length_list = []
        for j in range(i):
            row_length = len(rows[j])
            row_length_list.append(row_length)
        
        idx = np.sum(row_length_list)
        end_idx = idx-1
        
        perimeter_bead_index_list.append(end_idx)
        perimeter_bead_index_list.append(idx)
    
    # last row
    for i in range((len(all_beads)-len(rows[-1])+1),len(all_beads)):
        perimeter_bead_index_list.append(i)

    # set up index list
    bead_index_list = list(np.arange(len(all_beads)))
    
    # separate perimeter indices
    centre_bead_index_list = set(bead_index_list) - set(perimeter_bead_index_list)
    
    # separate corner indices
    n = len(rows[0])
    a = int((len(rows)-1)/2) - 1
    
    mid_row_idx = int((len(rows)-1)/2)
    
    first_mid_corner_idx = int(n + (a*n + (a**2)/2 + a/2))
    second_mid_corner_idx = first_mid_corner_idx + len(rows[mid_row_idx]) - 1
    
    corner_bead_index_list = np.array([0, n-1,
                                       first_mid_corner_idx, second_mid_corner_idx,
                                       bead_index_list[-n], bead_index_list[-1]])
    
    perimeter_bead_index_list = set(perimeter_bead_index_list) - set(corner_bead_index_list)

    return perimeter_bead_index_list, corner_bead_index_list, centre_bead_index_list

def nearest_neighbours(rows, all_beads):
    """
    This function returns the indices of, and the distances to, the nearest
    neighbours of each bead, given the rows and list of beads. This
    configuration will be maintained throughout sedimentation. 
    """
    # create dataframe containing all the bead positions
    position_df = pd.DataFrame(all_beads, columns=['x','y','z'])
    
    # calculate distances between each of the beads and store in dataframe
    distances = pdist(position_df, metric='euclidean')
    sq_dist = squareform(distances)
    sq_dist_df = pd.DataFrame(sq_dist, index=position_df.index,
                              columns=position_df.index)
    
    bead_index_dict = {}
    distances_dict = {}
    
    # sort the distances into ascending order and store indices in a separate df
    for bead in sq_dist_df.index:
        bead_distances_sorted = sq_dist_df.loc[bead].sort_values(ascending=True)
        bead_index_dict[bead] = bead_distances_sorted.index
        distances_dict[bead] = bead_distances_sorted.values

    bead_index_df = pd.DataFrame.from_dict(bead_index_dict).T
    bead_distances_df = pd.DataFrame.from_dict(distances_dict).T
        
    return bead_index_df, bead_distances_df

#%% STRETCHING FORCE PER BEAD
def spring(A,B):
    """
    This function returns the stretching force on bead A as a result of the 
    position of bead B. The returned forces have units of k, the stretching
    modulus.
    """
    AB = np.subtract(B,A)
    r = np.linalg.norm(AB)
    
    # prefactor in analytical derivative of stretching potential 
    prefactor = (r - r0) / r
    
    Fx = prefactor * AB[0]
    Fy = prefactor * AB[1]
    Fz = prefactor * AB[2]
    
    return -1*np.array([Fx,Fy,Fz])

def stretching_force_on_bulk_bead(positions):
    """
    This function returns the total stretching force on particle I given the
    positions of itself and 6 neighbours which it's connected to: J,K,L,M,N,O.
    The returned stretching force has units of k, the stretching modulus.
    """
    I = positions[0]
    J = positions[1]
    K = positions[2]
    L = positions[3]
    M = positions[4]
    N = positions[5]
    O = positions[6]
    
    FIJ = spring(I,J)
    FIK = spring(I,K)
    FIL = spring(I,L)
    FIM = spring(I,M)
    FIN = spring(I,N)
    FIO = spring(I,O)
    
    arr = np.array([FIJ, FIK, FIL, FIM, FIN, FIO])
    return sum(arr)

def stretching_force_on_edge_bead(positions):
    """
    This function returns the stretching forces on particle I given the
    positions of itself and four neighbours which it is connected to: J,K,L,M.
    The returned stretching force has units of k, the stretching modulus.
    """
    I = positions[0]
    J = positions[1]
    K = positions[2]
    L = positions[3]
    M = positions[4]

    FIJ = spring(I,J)
    FIK = spring(I,K)
    FIL = spring(I,L)
    FIM = spring(I,M)
    
    arr = np.array([FIJ,FIK,FIL,FIM])
    return sum(arr)

def stretching_force_on_corner_bead(positions):
    """
    This function returns the stretching forces on particle I given the
    positions of itself and three neighbours which it is connected to: J,K,L.
    The returned stretching force has units of k, the stretching modulus.
    """
    I = positions[0]
    J = positions[1]
    K = positions[2]
    L = positions[3]
    
    FIJ = spring(I,J)
    FIK = spring(I,K)
    FIL = spring(I,L)
    
    arr = np.array([FIJ,FIK,FIL])
    return sum(arr)

#%% TOTAL FORCES
def compute_stretching_forces(positions):
    """
    This function calculates the total stretching force upon each bead in the
    lattice by using the bead index to determine whether it lies in the middle
    or at the edge of the lattice and calling the appropriate stretching force
    function. 
    
    This function must be given a list of positions of the beads at which to 
    determine the total stretching force. These positions should be given as an
    array of 3-vectors. Note: it uses the initial positions to determine
    connectivity, rather than the given positions, and therefore requires the
    initial bead positions and rows as arrays of 3-vectors.
    """
    # stretching modulus - Swan et al.
    k = (np.sqrt(3)/2) * E * h 
    
    if SHEET_TYPE == 'hexagon':
        initial_rows, initial_all_beads = initial_hexagon_positions(edge_length=EDGE_LENGTH)
    
    elif SHEET_TYPE == 'fibre':
        initial_rows, initial_all_beads = initial_fibre_positions(fibre_length=FIBRE_LENGTH,
                                                                  fibre_width=FIBRE_WIDTH)
        
    assert len(positions) == len(initial_all_beads)

    # separate the indices of corner and perimeter beads from the bulk
    perimeter_idx_list, corner_idx_list, centre_idx_list = perimeter_separation(initial_rows,
                                                                                initial_all_beads)
    
    # determine each bead's closest neighbours by index
    bead_index_df, bead_distances_df = nearest_neighbours(initial_rows,
                                                          initial_all_beads)
    
    # set up force array 
    force_array = np.zeros((len(positions),3))

    # for the indices of centre beads, go into the bead index dataframe to 
    # find which are the nearest neighbours that it is directly connected to,
    # then use these indices in selecting which beads to use in the stretching
    # force calculation (take its nearest 6 neighbours - the dataframe includes
    # the bead's own index, so take up to index 7)
    for idx in centre_idx_list:
        nearest_neighbour_indices = bead_index_df.iloc[idx].iloc[0:7]
        
        nearest_neighbour_positions = []
        for jdx in nearest_neighbour_indices:
            position = positions[jdx]
            nearest_neighbour_positions.append(position) #includes itself
        
        stretching_force = stretching_force_on_bulk_bead(nearest_neighbour_positions)
        force_array[idx] = stretching_force
        
    # repeat for generic perimeter beads (only take the nearest 4 neighbours)
    for idx in perimeter_idx_list:
        nearest_neighbour_indices = bead_index_df.iloc[idx].iloc[0:5]
        
        nearest_neighbour_positions = []
        for jdx in nearest_neighbour_indices:
            position = positions[jdx]
            nearest_neighbour_positions.append(position)
    
        stretching_force = stretching_force_on_edge_bead(nearest_neighbour_positions)
        force_array[idx] = stretching_force
        
    # repeat for corner beads (only take the nearest 3 neighbours)
    for idx in corner_idx_list:
        nearest_neighbour_indices = bead_index_df.iloc[idx].iloc[0:4]
        
        nearest_neighbour_positions = []
        for jdx in nearest_neighbour_indices:
            position = positions[jdx]
            nearest_neighbour_positions.append(position)
    
        stretching_force = stretching_force_on_corner_bead(nearest_neighbour_positions)
        force_array[idx] = stretching_force
        
    return k * force_array

def compute_bending_forces(positions):
    """
    This function calculates the total bending force that each bead is subject
    to by calling the bending force function with the unit cell positions. The
    forces are associated with a particular bead index, and then summed
    up for each bead at the end.
    
    This function must be given a list of positions of the beads at which to 
    determine the total bending forces. These positions should be given as an
    array of 3-vectors. Note: it uses the initial positions to determine
    connectivity, rather than the given positions, and therefore requires the
    initial bead position function to also produce an array of 3-vectors.
    """
    # continuum bending rigidity - Swan et al.
    κ = E*h**3 / (np.sqrt(3) * 12 * (1-NU**2)) 
    
    if SHEET_TYPE == 'hexagon':
        initial_rows, initial_all_beads = initial_hexagon_positions(edge_length=EDGE_LENGTH)
    
    elif SHEET_TYPE == 'fibre':
        initial_rows, initial_all_beads = initial_fibre_positions(fibre_length=FIBRE_LENGTH,
                                                                  fibre_width=FIBRE_WIDTH)    
        
    assert len(positions) == len(initial_all_beads)
    
    # separate the indices of corner and perimeter beads from the bulk
    perimeter_idx_list, corner_idx_list, centre_idx_list = perimeter_separation(initial_rows,
                                                                                initial_all_beads)
    
    # determine each beads closest neighbours by index
    bead_index_df, bead_distances_df = nearest_neighbours(initial_rows,
                                                          initial_all_beads)
    
    # set up empty arrays to accept the indices that comprise the unit cells
    unit_cell_index_lists = []
    
    counter = 0
    
    # select only internal springs (indices of only centre beads)
    for idx in centre_idx_list:
        neighbour_indices = bead_index_df.iloc[idx].iloc[1:7]
        n_idx_set = set(neighbour_indices)
        
        # for each nearest neighbour of this centre bead, select the bead
        # indices that it shares with the centre bead
        
        for jdx in neighbour_indices:
            # for each neighbour, check its other neighbours
            if jdx in corner_idx_list:
                neighbour_of_neighbour_indices = bead_index_df.iloc[jdx].iloc[1:4]
            
            elif jdx in perimeter_idx_list:
                neighbour_of_neighbour_indices = bead_index_df.iloc[jdx].iloc[1:5]
            
            elif jdx in centre_idx_list:
                neighbour_of_neighbour_indices = bead_index_df.iloc[jdx].iloc[1:7] 
                
            else:
                raise IOError("Bead must belong to one of these index lists") 
                
            # neighbour's neighbour's index list
            n_n_idx_set = set(neighbour_of_neighbour_indices)
                
            # where the beads have two common neighbours there is a unit cell
            shared_neighbour_indices = set(n_n_idx_set.intersection(n_idx_set))
            shared_neighbour_indices_list = [i for i in neighbour_of_neighbour_indices if
                                             i in shared_neighbour_indices]
            
            # the two should only ever share two neighbours and together, these
            # four beads comprise a unit cell
            assert len(shared_neighbour_indices) == 2
            
            # put the resulting indices into the empty unit cell array like A,B,C,D
            # where A and C are the original bead and its direct neighbour that
            # we compare it to 
            unit_cell_index_lists.append(np.array([idx, shared_neighbour_indices_list[0], 
                                                   jdx, shared_neighbour_indices_list[1]])) 
            
            counter += 1
                
    # sorting the dataframe by increasing index to find duplicates
    unit_cell_index_lists_sorted = [np.sort(unit_cell_index_lists[arr]) for
                                    arr in range(len(unit_cell_index_lists))]
    
    unit_cell_indices_df = pd.DataFrame(unit_cell_index_lists, columns = ['A','B','C','D'])
    
    # we only want to select a spring once, so we need to remove repeat incidences:
    unit_cell_indices_df_sorted = pd.DataFrame(unit_cell_index_lists_sorted)
    unit_cell_indices_df_sorted = unit_cell_indices_df_sorted.drop_duplicates()
    
    unit_cell_indices_df = unit_cell_indices_df.reindex(unit_cell_indices_df_sorted.index)
    
    # resetting indices to not confuse iloc and loc
    unit_cell_indices_df = unit_cell_indices_df.reset_index(drop=True)
    
    # set up empty array to accept the bending force on each bead by index
    force_array = np.zeros((len(positions),3))
    
    # find the actual location of these beads using the indices
    for idx in list(unit_cell_indices_df.index):
        indices = unit_cell_indices_df.loc[idx]
        
        A = positions[indices['A']]
        B = positions[indices['B']]
        C = positions[indices['C']]
        D = positions[indices['D']]
        
        # use imported function from FlexibleSheet_helper
        F_A, F_B, F_C, F_D = unit_cell_bending_force(A,B,C,D)
   
        force_array[indices['A']] += F_A
        force_array[indices['B']] += F_B
        force_array[indices['C']] += F_C
        force_array[indices['D']] += F_D        
        
    return κ * force_array

def compute_gravitational_forces(positions):
    """
    Returns the gravitational force vectors on the beads. Note, buoyancy has
    been ignored throughout this study, but can easily be implemented for real
    scenarios. 
    """
    bead_mass = M/len(positions)
    acceleration_due_to_gravity = np.array([0,0,-9.81])
    bead_weight = bead_mass * acceleration_due_to_gravity
    
    weights = []
    for i in range(len(positions)):
        weights.append(bead_weight)

    return np.array(weights)

#%% MOBILITY
def Mij_block(r_i, r_j):
    """
    The mobility matrix is composed on NxN blocks of Mij, which are 3x3 matrices.
    This function takes in position vectors for i and j and returns the Mij
    for those two beads.
    """
    # in the form of Mij (i not equal j), require the square of the bead radius and 1/8pi mu
    a2 = R**2
    prefactor = 1/(8*np.pi*MU)
    
    # find the vector between the positions
    r = np.subtract(r_j, r_i)
    
    # need the length of the vector
    r2 = np.dot(r,r)
    r_norm = np.sqrt(r2)
    
    # calculate Mij blocks: 3x3 matrices
    if r_norm == 0:
        # if the distance between the beads is zero, it is the self mobility problem
        Mij = 1/(6 * np.pi * MU * R) * np.eye(3)
    
    else:
        # else it must different particles so use full expression
        Mij = (1 + (2*a2)/(3*r2))*np.eye(3) + (1 - (2*a2)/r2)*np.outer(r,r)/r2
        Mij = (prefactor/r_norm)*Mij
    
    return Mij

#%% VELOCITIES
def bead_velocities(positions):
    """
    Given a set of positions, this function should return the velocities of the
    beads by solving the mobility problem according to the forces that each
    bead is experiencing.
    """
    N = len(positions)
    
    # set up the 3N velocity vector
    velocities = np.zeros((N,3))
    
    # calculate all forces
    bending_forces = compute_bending_forces(positions)
    stretching_forces = compute_stretching_forces(positions)
    gravitational_forces = compute_gravitational_forces(positions)
    
    # add together the forces
    total_forces = np.zeros((N,3))
    for i in range(N):
        for j in range(3):
            total_forces[i][j] = bending_forces[i][j] + gravitational_forces[i][j] + stretching_forces[i][j]
        
    # calculate velocities
    for i in range(N):
        # starts with particle A, calculates its velocity, moves to B, etc.
        for j in range(N):
            # calculate the velocity of that particle based on the 3x3 Mij 
            # blocks that describe its mobility based on the positions of other
            # particles, by multiplication with the total forces on the particle
            Mij = Mij_block(positions[i], positions[j])
            velocities[i] += np.dot(Mij, total_forces[j]) # dot with other beads, j
            
    return velocities

#%% NUMERICAL INTEGRATION
def subsequent_positions(initial_all_beads):
    """
    Numerical integration to find subsequent locations of beads based upon the
    previous positions and forces.
    """
    number_of_steps = int(T/H)
    N = len(initial_all_beads)
    
    # establish empty arrays
    list_of_positions = np.zeros((number_of_steps,N,3))
    list_of_velocities = np.zeros((number_of_steps,N,3))
    
    # x_0, v_0
    list_of_positions[0] = initial_all_beads
    
    initial_velocities = bead_velocities(initial_all_beads)
    list_of_velocities[0] = initial_velocities    
    
    # iterate over a small timestep. At each stage, calculate new positions and 
    # feed these into the functions to find the subsequent velocities and repeat
    for i in tqdm(range(number_of_steps-1)):
        list_of_positions[i+1] = list_of_positions[i] + list_of_velocities[i]*H
        list_of_velocities[i+1] = bead_velocities(list_of_positions[i+1])
    
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(list_of_positions, f)
            
    return np.array(list_of_positions)

#%% MAIN
def main():
    """
    """
    # establish lattice
    if SHEET_TYPE == 'hexagon':
        initial_rows, initial_positions = initial_hexagon_positions(edge_length=EDGE_LENGTH)
    
    elif SHEET_TYPE == 'fibre':
        initial_rows, initial_positions = initial_fibre_positions(fibre_length=FIBRE_LENGTH,
                                                                  fibre_width=FIBRE_WIDTH)    
    
    # save setup diagram
    plot_initial_lattice(initial_positions)
    
    # applying rotation matrix to positions
    axis = [0, 1, 0] #axis of rotation: y-axis
    tilted_initial_positions = []
    for bead in initial_positions:
        new_pos = np.dot(rotation_matrix(axis,TILT), bead)
        tilted_initial_positions.append(new_pos)
    
    # generating and saving subsequent positions
    subsequent_positions(tilted_initial_positions)
    
    return None

#%%
if __name__ == '__main__':
    main()
    