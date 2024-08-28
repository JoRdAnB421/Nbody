import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

    
def ReadData(dataPath):
    with open(dataPath, 'r') as f:

        v = []
        masses = []
        pos = []
        times = []
        while True:
            try:
                N = int(f.readline())
            except:
                break
            m_i, pos_i, vel_i = [], [], []
            part = []
            times.append(float(f.readline()))

            for i in range(N):

                part.append(f.readline())
                part[i] = part[i].replace('\n', '')
                part[i] = part[i].split()

                m_i.append(float(part[i][0]))
            
                pos_i.append(part[i][1:4])
                vel_i.append(part[i][4:7])

                #print(pos_i)
                for j in range(3): # 3D system
                
                    pos_i[i][j] = float(pos_i[i][j])
                    vel_i[i][j] = float(vel_i[i][j])

            masses.append(m_i)
            pos.append(pos_i)
            v.append(vel_i)

    return v, np.asarray(pos), masses, times

_, pos,_,_, = ReadData('out.out')

print(pos[0])



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# lim=1e-3
# ax.set_xlim([-lim, lim])
# ax.set_ylim([-lim, lim])


# ax.view_init(elev=90, azim=0)


# Initialize empty lines for the traces
trail1 = []
trail2 = []
trail3 = []

# Initialize empty line objects for the trails
line1, = ax.plot([], [], [], '-', color='tab:blue')
line2, = ax.plot([], [], [], '-', color='tab:orange')
line3, = ax.plot([], [], [], '-', color='tab:green')
point1, = ax.plot([], [], [], 'o')
point2, = ax.plot([], [], [], 'o')
point3, = ax.plot([], [], [], 'o')

ax.legend()
ax.grid(False)
trail_length = 20

def update(frame):
    # Update the trace of the components
    # Update the trail for point 1
    trail1.append([pos[frame, 0, 0], pos[frame, 0, 1], pos[frame, 0, 2]])
    if len(trail1) > trail_length:
        trail1.pop(0)
    
    # Update the trail for point 2
    trail2.append([pos[frame, 1, 0], pos[frame, 1, 1], pos[frame, 1, 2]])
    if len(trail2) > trail_length:
        trail2.pop(0)
    
    # Update the trail for point 3
    trail3.append([pos[frame, 2, 0], pos[frame, 2, 1], pos[frame, 2, 2]])
    if len(trail3) > trail_length:
        trail3.pop(0)

    # Convert trails to numpy arrays for easy plotting
    trail1_np = np.array(trail1)
    trail2_np = np.array(trail2)
    trail3_np = np.array(trail3)

    # Update the line data with the current trails
    line1.set_data(trail1_np[:, 0], trail1_np[:, 1])
    line1.set_3d_properties(trail1_np[:, 2])

    line2.set_data(trail2_np[:, 0], trail2_np[:, 1])
    line2.set_3d_properties(trail2_np[:, 2])

    line3.set_data(trail3_np[:, 0], trail3_np[:, 1])
    line3.set_3d_properties(trail3_np[:, 2])

    point1.set_data(pos[frame, 0, 0], pos[frame, 0, 1])
    point1.set_3d_properties(pos[frame, 0, 2])

    point2.set_data(pos[frame, 1, 0], pos[frame, 1, 1])
    point2.set_3d_properties(pos[frame, 1, 2])

    point3.set_data(pos[frame, 2, 0], pos[frame, 2, 1])
    point3.set_3d_properties(pos[frame, 2, 2])

    x_center = (pos[frame, 0,0] + pos[frame, 1, 0] + pos[frame, 2,0]) / 3
    y_center = (pos[frame, 0,1] + pos[frame, 1,1] + pos[frame, 2,1]) / 3
    z_center = (pos[frame, 0,2] + pos[frame, 1,2] + pos[frame, 2,2]) / 3
    min_margin_x = np.min([pos[frame, 0,0] , pos[frame, 1, 0] , pos[frame, 2,0]]) -10
    min_margin_y = np.min([pos[frame, 0,1] , pos[frame, 1, 1] , pos[frame, 2,1]]) -10
    min_margin_z = np.min([pos[frame, 0,2] , pos[frame, 1, 2] , pos[frame, 2,2]]) -10
    
    max_margin_x = np.max([pos[frame, 0,0] , pos[frame, 1, 0] , pos[frame, 2,0]]) +10
    max_margin_y = np.max([pos[frame, 0,1] , pos[frame, 1, 1] , pos[frame, 2,1]]) +10
    max_margin_z = np.max([pos[frame, 0,2] , pos[frame, 1, 2] , pos[frame, 2,2]]) +10
    
    ax.set_xlim([min_margin_x, max_margin_x])
    ax.set_ylim([min_margin_y, max_margin_y])
    ax.set_zlim([min_margin_z, max_margin_z])
    ax.figure.canvas.draw()
    # return line1, line2, point1, point2, line3, point3
    return point1, point2, point3

ani = FuncAnimation(fig, update, frames=len(pos), blit=True)

plt.show()
