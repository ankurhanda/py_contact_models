from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv

import sys
sys.path.insert(0, '/home/ankur/workspace/code/py_contact_models/')

from timestepping import quaternions as quat
from plotting.plot_utils import sphere, cylinder
import csv
import matplotlib, time

states_np = []

with open('exp_gripper.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        st_np = np.asarray(row)
        states_np.append(st_np.astype(np.float))

states_np = np.array(states_np)
print(states_np.shape)

m = 60
r = 0.05
l = 0.005

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=30, azim=340)
plt.ion()
plt.axis('off')

# plot limits
lims = np.array([-3, 3, -3, 3, 0, 4])*r
lims[5] = 4*r + 2*l

x_s, y_s, z_s = sphere(m-1)

x_vec = np.reshape(x_s, (-1, 1))*r
y_vec = np.reshape(y_s, (-1, 1))*r
z_vec = np.reshape(z_s, (-1, 1))*r

pts_sphere = np.hstack((x_vec, y_vec, z_vec))

q0 = states_np[:, 0]
Y  = np.matmul(pts_sphere , quat.quat2mat(states_np[3:7, 0])) + q0[0:3].T 
x_s = Y[:, 0].reshape(m, m)
y_s = Y[:, 1].reshape(m, m)
z_s = Y[:, 2].reshape(m, m)

# Gripper (palm) mesh
xg = 0.050 * np.array([[0, 0, 0, 0, 0], [1, -1, -1,  1, 1], [1, -1, -1,  1,  1], [0, 0, 0, 0, 0]])
yg = 0.075 * np.array([[0, 0, 0, 0, 0], [1,  1, -1, -1, 1], [1,  1, -1, -1,  1], [0, 0, 0, 0, 0]])
zg =    2*l* np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

# Wrist 
[x_cyl,y_cyl,z_cyl] = cylinder([0.025, 0.025], 26)
z_cyl[1,:] = 4*r + 2*l + 0.02
z_cyl_copy = np.copy(z_cyl)

#Fingers 
xf = 0.010*np.array([[1, -1, -1,  1,  1], [1, -1, -1,  1,  1]])
yf = l*np.array([[1,  1, -1, -1,  1],[1,  1, -1, -1,  1]])
zf = 0.150*np.array([[ 0,  0,  0,  0,  0],  [1,  1,  1,  1,  1]])

# Plane
X, Y = np.meshgrid(np.linspace(-0.15, 0.15, 2), np.linspace(-0.15, 0.15, 2))

ax.set_xlim(lims[0], lims[1])
ax.set_ylim(lims[2], lims[3])
ax.set_zlim(lims[4], lims[5])
plt.show()

surf_sphere = []
surf_gripper = []
surf_finger1 = []
surf_finger2 = []
surf_wrist = []
plane_surface = []

for time_step in range(0, states_np.shape[1]):

    q_i = states_np[:, time_step]
    quat_i = q_i[3:7]

    Y  = np.matmul(pts_sphere , quat.quat2mat(quat_i)) + q_i[0:3].T 
    x_s = Y[:, 0].reshape(m, m)
    y_s = Y[:, 1].reshape(m, m)
    z_s = Y[:, 2].reshape(m, m)

    # Sphere 
    if time_step > 0:
        surf_sphere.remove()
    
    surf_sphere = ax.plot_surface(x_s, y_s, z_s, 
                                  color=[0.1, 0.5, 0.5, 0.4],
                                linewidth=2, antialiased=True)

    #Fingers
    if time_step > 0:
        surf_finger1.remove()

    surf_finger1 = ax.plot_surface(xf, yf + q_i[7] + l, zf + q_i[11], color=[0.5, 0.5, 0.5, 0.4],
                       linewidth=2, antialiased=True)

    if time_step > 0:
        surf_finger2.remove()

    surf_finger2 = ax.plot_surface(xf, yf + q_i[8] - l, zf + q_i[11], color=[0.5, 0.5, 0.5, 0.4],
                       linewidth=2, antialiased=True)

    # Gripper 
    if time_step > 0:
        surf_gripper.remove()
    surf_gripper = ax.plot_surface(xg + q_i[9], yg + q_i[10], zg + q_i[11] + 0.15, 
                                   color=[0.5, 0.5, 0.5, 0.2], linewidth=1, antialiased=True)
 
    # Wrist 
    if time_step > 0:
        surf_wrist.remove()
    z_cyl[0,:] = z_cyl_copy[0,:] + q_i[11] + 0.15 + 2*l
    surf_wrist = ax.plot_surface(x_cyl, y_cyl, z_cyl, color=[0.5, 0.5, 0.5, 0.4],
                       linewidth=1, antialiased=True)

    # Plane 
    if time_step > 0:
        plane_surface.remove()

    X, Y = np.meshgrid(np.linspace(lims[0], lims[1], 2), np.linspace(lims[0], lims[1], 2))
    plane_surface = ax.plot_surface(X, 
                    Y,
                    np.array([[0, 0], [0, 0]]),
                    color=[0.5, 0.5, 0.5, 0.2], linewidth=1, antialiased=True)

    fig.savefig("gripper_{:04d}.png".format(time_step))
    fig.canvas.draw()                     
    fig.canvas.flush_events()
    time.sleep(0.01)
  
plt.ioff()
plt.show()