import numpy as np
import matplotlib.pyplot as plt
from numba import njit

G=1
M=1
vy0 = 0.8*np.sqrt(G*M/4)
#np.sqrt((2*G*M*10**(-7)/(2*(1+1**(-7)))))

r0 = np.array([1,0, 0.0, vy0], float)
tpoints = []
xpoints = []
ypoints = []
rpoints = []
h_array = []
t_totpoints = []
ratio_points = []
Brange = [0.5,3, 6,10]
Arange = [0.5, 3, 6, 10]
delta_Epoints = []
E0= 0.5*vy0**2-(G*M/(4))
t0 = 0
h0 = 0.1
h = h0
d = 1e-7#tolerance per unit time

def f(r, A, B):
    x = r[0]
    vx = r[1]
    y = r[2]
    vy = r[3]
    rbh = np.sqrt(x**2+y**2)
    vbh = np.sqrt(vx**2+vy**2)
    vdfx = (-A/(vbh**3+B))*vx
    vdfy = (-A/(vbh**3+B))*vy
    fx = vx
    fvx = -(G*M*x)/(4*(rbh)**3)+vdfx
    fy = vy
    fvy = -(G*M*y)/(4*(rbh)**3)+vdfy
    return np.array([fx, fvx, fy, fvy], float)

def RK4(r,t, step, A, B):
    k1 = step*f(r, A, B)
    k2 = step*f(r+0.5*k1, A, B)
    k3 = step*f(r+0.5*k2, A, B)
    k4 = step*f(r+k3, A, B)
    return (k1+2*k2+2*k3+k4)/6


for A in Arange:
    for B in Brange:
        ratio = B/A
        r = r0.copy()
        t=0
        while np.sqrt(r[0]**2+r[2]**2)>=10**(-7):
            
            del_r_h = RK4(r,t,h, A, B)
            del_r_hh = RK4(r+del_r_h,t+h,h, A, B)
            del_double = del_r_h + del_r_hh
            r_2h = RK4(r, t, 2*h, A, B)
            
            error = np.sqrt((del_double[0]-r_2h[0])**2+(del_double[2]-r_2h[2])**2) # we only care about the error for r=sqrt(x^2+y^2). The velocities are quantities only used at our convenience
            rho = (h*d)/error
            if rho >=1:
                h_array.append(h)
                tpoints.append(t)
                xpoints.append(r[0])
                ypoints.append(r[2])
                v = np.sqrt(r[1]**2+r[3]**2)
                rmag = np.sqrt(r[0]**2+r[2]**2)
                rpoints.append(rmag)
                #E = 0.5*v**2-(G*M/(4*rmag))
                #delta_Epoints.append(abs(E-E0)/abs(E0))
                t += 2*h
                h *= min((rho)**0.25,2)
                r += del_double
                
            else:
                h *= rho**(0.25)
        t_totpoints.append(t)
        ratio_points.append(ratio)
    
print(tpoints)
print(vy0)

#fig, ax = plt.subplots(figsize = (6.5,4))
#ax.plot(np.array(xpoints, float), np.array(ypoints, float), 'b')
#ax.set_xlabel('X/ 100 parsecs')
#ax.set_ylabel('Y/ 100 parsecs')
#ax.set_title('Motion of black hole with friction')

fig, ax2 = plt.subplots(figsize = (6.5,4))
ax2.plot(ratio_points, t_totpoints, 'b')
ax2.set_xlabel("B/A")
ax2.set_ylabel("Time")
ax2.set_title("Time taken to reach the Schwarzschild radius for different initial conditions of A and B")


# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
plt.tight_layout()
plt.savefig("timetoreachrS.png")
plt.show()
