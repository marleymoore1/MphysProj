#Imports
import numpy as np
import matplotlib.pyplot as plt
#%%
#Defining global variables so that it is easy to change the problem parameters
G = -9.81 #acceleration due to gravity m/s^2
SIGMA = 2700 #sphere density kg/m^3
RHO = 1.2 #fluid density kg/m^3
MU = 1.825e-2 #dynamic viscosity of the fluid in kg m^-1 s^-1

def maximum_radius(g, sigma, rho, mu):
    #for the slow flow equations to hold, radius must be smaller than a maximum
    A = 9*mu**2
    B = 20*np.abs(g)*(sigma-rho)*rho
    
    r_max = np.cbrt(A/B)
    return r_max

R_MAX = maximum_radius(G, SIGMA, RHO, MU) #slow flow holds below this radius
R = R_MAX-R_MAX/1000 #radius of sphere in m
# R = 0.0000001

M = (4/3*np.pi*R**3) * SIGMA #calculates the mass of the sphere in kg
V_MAX = 1/10 * MU/(RHO*R) #above this velocity, slow flow equations do not hold
V_T = 2/9 * (R**2) * G/MU * (SIGMA-RHO) #terminal velocity reached by sphere

if np.abs(V_T) > np.abs(V_MAX):
    raise Exception("Terminal velocity reached breaches the slow flow ragime")

x_0=0 #starting position in m
v_0=0 #starting velocity in m

T=1 #total sedimentation time step in s
H=0.003 #default timestep for numerical integration in s
#%%
#Comparing analytical solutions to positions of sphere compared to those obtained via numerical integration.
def weight(g, sigma, r):
    #calculates weight force
    return 4/3 * np.pi * r**3 * sigma * g

def buoyancy(g, rho, r):
    #calculates buoyancy force
    return 4/3 * np.pi * r**3 * rho * g

def drag_force(velocity, mu, r):
    #calculates drag force
    return 6 * mu * velocity * r * np.pi

def step_number(t, h):
    #calculates the number of numerical integration steps from global parameters
    number_of_steps = int(t/h) #Rounds down
    return number_of_steps

def time(t, h):
    #gives an array of times for plotting
    number_of_steps = step_number(t, h)
    times = np.arange(0, number_of_steps*h, h)
    return times

def analytical_integration(t, h, g, sigma, rho, r, m, mu):
    #calculates the positions of the sphere through time from the analytical solution
    times = time(t, h)

    A = 6*np.pi*mu*r/m
    B = g*(1 - rho/sigma)

    positions = (B/(A**2)) * (np.exp(-A*times)-1) + (B/A)*times
    return positions

def numerical_integration(t, h, g, sigma, rho, r, m, mu):
    #Euler method of iterating through small timesteps to give position and velocity arrays
    number_of_steps = step_number(t, h)

    x = np.zeros(number_of_steps)
    v = np.zeros(number_of_steps)

    weigh = weight(g, sigma, r)
    buoy = buoyancy(g, rho, r)

    x[0] = x_0
    v[0] = v_0
    # a[0] = (weight(g, sigma, r)-buoyancy(g, rho, r))/m

    for i in range(number_of_steps-1):
        a = (weigh - buoy - drag_force(v[i], mu, r))/m
        x[i+1] = x[i] + v[i]*h + 0.5*a*h**2
        v[i+1] = v[i] + a*h
    return x 

#%%
def plot_alongside(t, h, g, sigma, rho, r, m, mu):
    #plots the positions from both methods side by side for comparison
    times = time(t, h)
    x_analyt = analytical_integration(t, h, g, sigma, rho, r, m, mu)
    x_nume  = numerical_integration(t, h, g, sigma, rho, r, m, mu)

    plt.plot(times, x_analyt, 'b', label='Analytical')
    plt.plot(times, x_nume, 'g', label='Numerical')
    
    plt.xlabel('time / $s$')
    plt.ylabel('displacement / $m$')
    plt.title('timestep = {0}, terminal velocity = {1: .3g} $m/s$'.format(h, V_T))
    plt.legend()
    plt.show()
    return None

plot_alongside(T, H, G, SIGMA, RHO, R, M, MU)

#%%
def error_difference(t, g, sigma, rho, r, m, mu, h_min, h_max):
    
    h_list = np.arange(h_min, h_max+h_min, h_min)
    err_list = []

    for h in h_list:

        x_analyt       = analytical_integration(t, h, g, sigma, rho, r, m, mu)
        x_nume         = numerical_integration(t, h, g, sigma, rho, r, m, mu)

        # mean_error = np.mean(np.abs(x_analyt[:] - x_nume[:]))
        final_error = np.abs(x_analyt[-1] - x_nume[-1])

        err_list = np.append(err_list, final_error)

    return err_list, h_list

def plot_error_difference(t, g, sigma, rho, r, m, mu, h_min, h_max):
    err_list, h_list =  error_difference(T, G, SIGMA, RHO, R, M, MU, h_min, h_max)

    plt.plot(h_list, err_list, 'k')
    plt.xlabel('timestep / $s$')
    plt.ylabel('error at end / $m$')
    plt.title('Final error for a range of timesteps')
    plt.show()

    return None

plot_error_difference(T, G, SIGMA, RHO, R, M, MU, h_min=0.001, h_max=0.05)

#%%









