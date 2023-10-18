import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

'''
site energy -> activation energy -> diffusion rate

site energy(E_i) = -N*(E_b)/2

activation energy(E_a) = E_0 + alpha * E_r
E_r = E_i(end) - E_i(start)

diffusion rate = f*exp(-E_a/(k*T)) (Arrhenius)
f is ~ 10^13 for most metals
'''

def get_site_energy(bond_energy, bond_num):
    return -bond_num * bond_energy/2

def get_activation_energy(e_start, e_end, alpha = 0.1, e0 = 0):
    e_reaction = e_end-e_start
    if e_reaction>=0:
        return e0 + (1+alpha)*e_reaction
    else:
        return e0 + alpha*e_reaction

def get_diffusion_rate(e_a, T=300, f=1E13):
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
    return f*np.exp(-e_a/(k_B*T))

'---------------------------------------------------------'

# function to initialize the lattice
'''
Initialize the lattice
1 : atom
0 : vacancy or vacumm
2 more layer to make vacuum
'''
def init_lattice(width, height):
    lattice = np.ones((height+2, width), dtype=int)
    lattice[0, :] = 0
    lattice[-1, :] = 0

    return lattice

# Function to draw lattice on prompt
def draw_lattice(lattice):
    height, width = lattice.shape
    for y in range(height):
        for x in range(width):
            if lattice[y, x] == 1:
                print('●', end = '')
            else:
                print(' ', end='')
        print('\n')

'''
function that finds every possible way
1. atom jumped from surface
2. return to previous position(jumped from surface)
3. other diffusion...
'''
def get_atoms_around_site(lattice, x, y, total=False):
    height, width = lattice.shape
    # calculate bond number
    if y == height-1:
        down = 0
    else:
        down = lattice[y+1, x]
    if y == 0:
        up = 0
    else:
        up = lattice[y-1, x]
    # PBC at left and right
    right = lattice[y, (x+1)%width]
    left = lattice[y, (x-1)%width]
    
    if total:
        return left+up+right+down
    else:
        return left, up, right, down

def find_candidate(lattice):
    global e_a
    global diffusion_rate
    height, width = lattice.shape
    candidate_table = []
    diffusion_table = []
    motion_table = []
    for x in range(width):
        for y in range(height):
            # find vacancy
            if lattice[y, x] == 0:
                left, up , right, down = get_atoms_around_site(lattice, x, y)
                neighbor = left+up+right+down
                if neighbor:
                    if left:
                        neighbor_of_left = get_atoms_around_site(lattice, (x-1)%width, y, True)
                        candidate_table.append((y, (x-1)%width))
                        motion_table.append(3)
                        diffusion_table.append(diffusion_rate[neighbor_of_left, neighbor-1])
                    if right:
                        neighbor_of_right = get_atoms_around_site(lattice, (x+1)%width, y, True)
                        candidate_table.append((y, (x+1)%width))
                        motion_table.append(1)
                        diffusion_table.append(diffusion_rate[neighbor_of_right, neighbor-1])
                    if up:
                        neighbor_of_up = get_atoms_around_site(lattice, x, y-1, True)
                        candidate_table.append((y-1, x))
                        motion_table.append(4)
                        diffusion_table.append(diffusion_rate[neighbor_of_up, neighbor-1])
                    if down:
                        neighbor_of_down = get_atoms_around_site(lattice, x, y+1, True)
                        candidate_table.append((y+1, x))
                        motion_table.append(2)
                        diffusion_table.append(diffusion_rate[neighbor_of_down, neighbor-1]) 
    
    return candidate_table, diffusion_table, motion_table

# KMC function
def diffuse_one_step(lattice, print_out=False):
    global time_elapsed
    cand, dif, motion = find_candidate(lattice)
    dif = np.array(dif)

    total_dif = np.sum(dif)
    
    # pick 1
    u = np.random.uniform(low=1e-6, high=1)
    u_time = np.random.uniform(low=1e-6, high=1)
    cum_dif = np.cumsum(dif)

    chosen_idx = np.argwhere(u*total_dif < cum_dif)[0][0]

    # print information
    if print_out:
        print(f'total_diff : {total_dif}')
        print(f'chosen : {cand[chosen_idx]} atom')   
    # print(f'motion : {motion[chosen_idx]}')
    # print(cand[chosen_idx])
    # print(motion[chosen_idx])

    # change the lattice
    x, y = cand[chosen_idx][0], cand[chosen_idx][1]

    # get motion
    if motion[chosen_idx] == 1:
        lattice[x, y] = 0
        lattice[x, (y-1+width)%width] = 1
        if print_out:
            print(f'go left')   
    if motion[chosen_idx] == 2:
        lattice[x, y] = 0
        lattice[x-1 , y] = 1
        if print_out:
            print(f'go up')   
    if motion[chosen_idx] == 3:
        lattice[x, y] = 0
        lattice[x, (y+1)%width] = 1
        if print_out:
            print(f'go right')   
    if motion[chosen_idx] == 4:
        lattice[x, y] = 0
        lattice[x+1 , y] = 1
        if print_out:
            print(f'go down')   
    
    # time update
    delta_t = -np.log(u_time)/total_dif
    # print(delta_t)
    time_elapsed += delta_t
'--------------------------------------------------------------------------------------------'
# Parameters
# Size of the lattice (height x width)
width = 100  
height = 10

# parameter for diffusion rate
'''
bond energy : 200kJ/mol ~~ 2.07 eV/particle
'''
bond_energy = 2.07 
temperature = 300
e0 = 0.1

# site energy, e_(bond number)
e_site = np.zeros(4)
for i in range(4):
    e_site[i] = get_site_energy(bond_energy, i)

# activation energy, e_a_(start to end)
e_a = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        e_a[i, j] = get_activation_energy(e_site[i], e_site[j], e0=e0)

# diffusion rate, rate_(start to end)
diffusion_rate = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        diffusion_rate[i, j] = get_diffusion_rate(e_a[i, j], temperature)
  
steps = 10000
time_elapsed = 0

if __name__ == "__main__":
    lattice = init_lattice(width, height)
    for i in range(1, steps+1):
        print(f'---------------- step {i} ---------------------')
        diffuse_one_step(lattice, True)
        draw_lattice(lattice)
        print(f'Time elapsed : {time_elapsed} s')
        time.sleep(0.4)
        
        os.system('clear')
        