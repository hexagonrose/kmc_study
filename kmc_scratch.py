import time
import os
import numpy as np

# Parameter tuning if you want
width = 100  # Size of the square lattice (height x width)
height = 10

surface_jump_rate = 0.1     # bond 3 -> 0
return_rate = 100            # bond 0 -> 3
surface_diffusion_rate = 30  # bond 2 -> 3

atom_radius = 0.4
steps = 1000
t = 0


# function to initialize the lattice
'''
Initialize the lattice
1 : atom
0 : vacancy or vacumm
'''
def init_lattice(width, height):
    lattice = np.ones((height+2, width), dtype=int)
    lattice[0, :] = 0
    lattice[-1, :] = 0

    return lattice

'''
function that finds every possible way
current state
1. atom jump at surface
2. return to previous position(jumped from surface)
3. atom jump from side
'''
def find_candidate(lattice):
    candidate_table = []
    diffusion_table = []
    motion_table = []
    for x in range(width):
        for y in range(height+2):
            # find atom(value 1)
            if lattice[y, x] == 1:
                # calculate bond number
                if y == height+1:
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
                bond_num = down + up + right + left

                # classify the atoms by bond_num
                # 0 bond num : jumped atom from surface
                if bond_num == 0:
                    candidate_table.append((y, x))
                    diffusion_table.append(return_rate)
                    if y == height+1 :
                        motion_table.append(2)
                    else :
                        motion_table.append(4)
                # 2 bond num : surface atom next to vacancy
                elif bond_num == 2:
                    candidate_table.append((y, x))
                    diffusion_table.append(surface_diffusion_rate)
                    if left == 0 :
                        motion_table.append(1)
                    else:
                        motion_table.append(3)
                # 3 bond num
                # 3-1. surface atom
                elif bond_num == 3:
                    candidate_table.append((y, x))
                    diffusion_table.append(surface_jump_rate)
                    if left == 0 :
                        motion_table.append(1)
                    elif up == 0 :
                        motion_table.append(2)
                    elif right == 0 :
                        motion_table.append(3)
                    else:
                        motion_table.append(4)
                
    
    return candidate_table, diffusion_table, motion_table


# KMC function
def diffuse_one_step(lattice, print_out=False):
    cand, dif, motion = find_candidate(lattice)
    dif = np.array(dif)

    total_dif = np.sum(dif)
    

    # pick 1
    u = np.random.uniform(low=1e-6, high=1)
    cum_dif = np.cumsum(dif)

    chosen_idx = np.argwhere(u*total_dif < cum_dif)[0][0]

    # print information
    # print(f'total_diff : {total_dif}')
    if print_out:
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


def draw_lattice(lattice):
    height, width = lattice.shape
    for y in range(height):
        for x in range(width):
            if lattice[y, x] == 1:
                print('●', end = '')
            else:
                print(' ', end='')
        print('\n')

# Parameter tuning if you want
width = 100  # Size of the square lattice (height x width)
height = 10

surface_jump_rate = 0.1     # bond 3 -> 0
return_rate = 100            # bond 0 -> 3
surface_diffusion_rate = 30  # bond 2 -> 3

atom_radius = 0.4
steps = 100
t = 0



if __name__ == "__main__":
    lattice = init_lattice(width, height)
    for i in range(1, steps+1):
        print(f'---------------- step {i} ---------------------')
        diffuse_one_step(lattice)
        draw_lattice(lattice)
        time.sleep(0.4)
        os.system('clear')
