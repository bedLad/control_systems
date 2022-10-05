import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import random

trials = 3
incl_angle = (np.pi/6) * 1.1
g = 9.78
mass_cart = 100

Kp = 300
Kd = 390
Ki = 5

trials_global = trials

# Generating random position for a falling cube
def set_x_ref (incl_angle):
    rand_x = random.uniform(0,120) # generating a random location along x-axis
    '''
    the first parameter defines the start of range which is 20+6.5 units above the cart
    so that the cart can catch it.
    '''
    rand_y = random.uniform(20+120*np.tan(incl_angle)+6.5, 40+120*np.tan(incl_angle)+6.5)
    return rand_x, rand_y

dt = 0.02
t0 = 0
t_end = 5
t = np.arange(t0, t_end+dt, dt)

F_g = -mass_cart*g

'''
PARAMETERS FOR THE TRAIN
this is diplacement of cart 
here is made a 2d array made of zeros with 
rows = no. of times a trial has been performed & 
col = len(t)
'''
displ_rail = np.zeros((trials, len(t))) 
v_rail = np.zeros((trials, len(t))) 
a_rail = np.zeros((trials, len(t))) 
pos_x_train = np.zeros((trials,len(t)))
pos_y_train = np.zeros((trials,len(t)))
e = np.zeros((trials,len(t)))
e_dot = np.zeros((trials,len(t)))
e_int = np.zeros((trials,len(t)))

pos_x_cube=np.zeros((trials,len(t)))
pos_y_cube=np.zeros((trials,len(t)))

F_ga_t = F_g*np.sin(incl_angle)  # Tangential component of gravity
init_pos_x = 120
init_pos_y = 120*np.tan(incl_angle) + 6.5
init_displ_rail = (init_pos_x**2 + init_pos_y**2)**(0.5)
init_vel_rail = 0
init_a_rail = 0

init_pos_x_global = init_pos_x # USed to determine dimensions of the animation window

trials_magn = trials
history = np.ones(trials)
while (trials > 0):
    # storing the random value generated 
    pos_x_cube_ref = set_x_ref(incl_angle)[0]
    pos_y_cube_ref = set_x_ref(incl_angle)[1]
    times = trials_magn - trials
    pos_x_cube[times] = pos_x_cube_ref
    pos_y_cube[times] = pos_y_cube_ref - g/2*t**2
    win = False
    delta = 1

    # implement PID for the train position
    for i in range(1,len(t)):
        
        # Insert the initial values into the beginning of predefined arrays
        if i == 1:
            displ_rail[times][0] = init_displ_rail
            pos_x_train[times][0] = init_pos_x
            pos_y_train[times][0] = init_pos_y
            v_rail[times][0] = init_vel_rail
            a_rail[times][0] = init_a_rail

        # compute horizontal errors
        e[times][i-1] = pos_x_cube_ref - pos_x_train[times][i-1]

        if i>1:
            e_dot[times][i-1] = (e[times][i-1] - e[times][i-2])/dt
            e_int[times][i-1] = e_int[times][i-2] + ((e[times][i-2] + e[times][i-1])/2)*dt
        
        if i == len(t)-1:
            e[times][-1] = e[times][-2]
            e_dot[times][-1] = e_dot[times][-2]
            e_int[times][-1] = e_int[times][-2]
        
        F_a = Kp*e[times][i-1] + Kd*e_dot[times][i-1] + Ki*e_int[times][i-1]
        F_net = F_a + F_ga_t
        a_rail[times][i] = F_net/mass_cart
        v_rail[times][i] = v_rail[times][i-1] + ((a_rail[times][i] + a_rail[times][i-1])/2)*dt
        displ_rail[times][i] = displ_rail[times][i-1] + ((v_rail[times][i] + v_rail[times][i-1])/2)*dt
        pos_x_train[times][i] = displ_rail[times][i]*np.cos(incl_angle)
        pos_y_train[times][i] = displ_rail[times][i]*np.sin(incl_angle) + 6.5

        # trying to catch the train
        if (pos_x_train[times][i] -5 < pos_x_cube[times][i] +3 and pos_x_train[times][i] +5 > pos_x_cube[times][i] -3) or win == True:
            if (pos_y_train[times][i] +3 < pos_y_cube[times][i] -2 and pos_y_train[times][i] + 8.5 > pos_y_cube[times][i]+2) or win == True:
                win = True
                '''
                following 5 lines determines cube being caught by th train and then moving along with it
                if statement calculates the distance b/w cube center to train center
                '''
                if delta == 1:
                    change = pos_x_train[times][i] - pos_x_cube[times][i]
                    delta = 0
                
                pos_x_cube[times][i] = pos_x_train[times][i] - change   # latest position of cube is updated
                pos_y_cube[times][i] = pos_y_train[times][i] + 5
    
    '''
    now after the cube is caught, new trial starts but the train remains at the same
    position where it caught the cube thanks to the following code
    '''
    init_displ_rail = displ_rail[times][-1]
    init_pos_x = pos_x_train[times][-1] + v_rail[times][-1]*np.cos(incl_angle)*dt
    init_pos_y = pos_y_train[times][-1] + v_rail[times][-1]*np.sin(incl_angle)*dt
    init_vel_rail = v_rail[times][-1]
    init_a_rail = a_rail[times][-1]
    history[times] = delta
    trials = trials-1


##########################  ANIMATION #########################
len_t = len(t)
frame_amount = len(t)*trials_global

def update_plot( num ):

    platform.set_data([pos_x_train[int(num/len_t)][num - int(num/len_t)*len_t] -3.1,\
    pos_x_train[int(num/len_t)][num - int(num/len_t)*len_t] +3.1],\
    [pos_y_train[int(num/len_t)][num - int(num/len_t)*len_t],\
    pos_y_train[int(num/len_t)][num - int(num/len_t)*len_t]])
    
    cube.set_data([pos_x_cube[int(num/len_t)][num - int(num/len_t)*len_t] -1,\
    pos_x_cube[int(num/len_t)][num - int(num/len_t)*len_t] +1],\
    [pos_y_cube[int(num/len_t)][num - int(num/len_t)*len_t],\
    pos_y_cube[int(num/len_t)][num - int(num/len_t)*len_t]])
    
    if trials_magn*len_t == num+1 and num > 0: 
        if sum(history)==0:
            success.set_text('Success')
        else:
            again.set_text('Failed')

    displ_rail_f.set_data(t[0:(num - int(num/len_t)*len_t)],
        displ_rail[int(num/len_t)][0:(num - int(num/len_t)*len_t)])
    
    v_rail_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        v_rail[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    a_rail_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        a_rail[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    e_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        e[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    e_dot_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        e_dot[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    e_int_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        e_int[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    return displ_rail_f, v_rail_f, a_rail_f, e_f, e_dot_f, e_int_f, platform, success, again, cube

fig = plt.figure(figsize=(16,9), dpi=120, facecolor=(0.8,0.8,0.8))
gs = gridspec.GridSpec(4,3)

# Create the main window
ax_main = fig.add_subplot(gs[0:3,0:2], facecolor=(0.9,0.9,0.9))
plt.xlim(0, init_pos_x_global)
plt.ylim(0, init_pos_x_global)
plt.xticks(np.arange(0, init_pos_x_global + 1, 10))
plt.yticks(np.arange(0, init_pos_x_global + 1, 10))
plt.grid(True)

rail = ax_main.plot([0, init_pos_x_global], [5, init_pos_x_global*np.tan(incl_angle) + 5], 'k', linewidth=6)
platform ,= ax_main.plot([],[], 'r', linewidth=18)
cube ,= ax_main.plot([],[], 'k', linewidth=14)

bbox_props_success=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw='1')
success=ax_main.text(40, 60, '', size='20', color='g', bbox=bbox_props_success)

bbox_props_again = dict(boxstyle='square', fc=(0.9,0.9,0.9), ec='r', lw='1')
again=ax_main.text(30, 60, '', size='20', color='r', bbox=bbox_props_again)

# Plot windows
ax1v = fig.add_subplot(gs[0,2], facecolor=(0.9,0.9,0.9))
displ_rail_f,=ax1v.plot([],[],'-b',linewidth=2,label='displ. on rails [m]')
plt.xlim(t0,t_end)
plt.ylim(np.min(displ_rail)-abs(np.min(displ_rail))*0.1,np.max(displ_rail)+abs(np.max(displ_rail))*0.1)
plt.grid(True)
plt.legend(loc='lower left',fontsize='small')

ax2v=fig.add_subplot(gs[1,2],facecolor=(0.9,0.9,0.9))
v_rail_f,=ax2v.plot([],[],'-b',linewidth=2,label='velocity on rails [m/s]')
plt.xlim(t0,t_end)
plt.ylim(np.min(v_rail)-abs(np.min(v_rail))*0.1,np.max(v_rail)+abs(np.max(v_rail))*0.1)
plt.grid(True)
plt.legend(loc='lower left',fontsize='small')

ax3v=fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
a_rail_f,=ax3v.plot([],[],'-b',linewidth=2,label='accel. on rails [m/s^2] = F_net/m_platf.')
plt.xlim(t0,t_end)
plt.ylim(np.min(a_rail)-abs(np.min(a_rail))*0.1,np.max(a_rail)+abs(np.max(a_rail))*0.1)
plt.grid(True)
plt.legend(loc='lower left',fontsize='small')

ax1h=fig.add_subplot(gs[3,0],facecolor=(0.9,0.9,0.9))
e_f,=ax1h.plot([],[],'-b',linewidth=2,label='horizontal error [m]')
plt.xlim(t0,t_end)
plt.ylim(np.min(e)-abs(np.min(e))*0.1,np.max(e)+abs(np.max(e))*0.1)
plt.grid(True)
plt.legend(loc='lower left',fontsize='small')

ax2h=fig.add_subplot(gs[3,1],facecolor=(0.9,0.9,0.9))
e_dot_f,=ax2h.plot([],[],'-b',linewidth=2,label='change of horiz. error [m/s]')
plt.xlim(t0,t_end)
plt.ylim(np.min(e_dot)-abs(np.min(e_dot))*0.1,np.max(e_dot)+abs(np.max(e_dot))*0.1)
plt.grid(True)
plt.legend(loc='lower left',fontsize='small')

ax3h = fig.add_subplot(gs[3,2],facecolor=(0.9,0.9,0.9))
e_int_f ,= ax3h.plot([],[],'-b',linewidth=2,label='sum of horiz. error [m*s]')
plt.xlim(t0,t_end)
plt.ylim(np.min(e_int)-abs(np.min(e_int))*0.1,np.max(e_int)+abs(np.max(e_int))*0.1)
plt.grid(True)
plt.legend(loc='lower left',fontsize='small')

pid_ani=animation.FuncAnimation(fig, update_plot, frames=frame_amount, interval=20, repeat=False, blit=True)
plt.show()
