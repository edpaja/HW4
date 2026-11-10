import numpy as np
import matplotlib.pyplot as plt

# dx/dt = 1-(b+1)*x+a*x**2*y
# dy/dt = b*x - a*x**2*y



a=1.0
b=3.0
x0 = 0
y0 = 0
r0 = [x0,y0]
t0 = 0

def f(r):
    x = r[0]
    y = r[1]
    fx = 1-(b+1)*x+a*x**2*y
    fy = b*x - a*x**2*y
    return np.array([fx, fy], float)

def modified_mid(r, interval, row):
    h = interval / row
    k = r + 0.5 * h * f(r)
    r = r + h * f(k)
    for repeat in range(row - 1):
        k += h * f(r)
        r += h * f(k)
    return 0.5 * (r + k + 0.5 * h * f(r))

H_0 = 20
H=H_0
t = [0]
rpoints = [np.array([0,0])]

def n_triangle(r, t, H):
    R1 = [modified_mid(r, H, 1)]
    for n in range(2, 9):
        R2 = [modified_mid(r, H, n)]
        for m in range(1, n):
            R2.append(R2[m - 1] + (R2[m - 1] - R1[m - 1]) /
                      ((n / (n - 1.0))**(2 * m) - 1))
        m=n-1
        epsilon = (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1)
        error = max(abs(epsilon))
        if error < H * 10**(-10):#setting error per unit time
            return R2[n - 1]
        R1 = R2 #move to the next row by shifting the triangles
    return None

def bs_method(r, t, H, tpoints, rpoints):
    r_H = n_triangle(r,t,H)
    if r_H is None:
        half = bs_method(r,t,H/2, tpoints, rpoints)
        return bs_method(half, t+H/2, H/2, tpoints, rpoints)
    tpoints.append(t+H)
    rpoints.append(r_H)
    return r_H

bs_method(r0,0,20,t, rpoints)
rpoints = np.array(rpoints)
t = np.array(t)
print(len(t))
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(t, rpoints[:,0], color="green", linestyle="-", label="x")
ax.plot(t, rpoints[:,0], color="green", marker=".", label="x")
ax.plot(t, rpoints[:,1], color="red", linestyle="-", label="y")
ax.plot(t, rpoints[:,1], color="red", marker=".", label="y")
ax.set(xlabel="time", xlim=(0, 20), ylabel="concentration",
       title="Brusselator model")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("/Users/edouardsavalle/Desktop/University/grad/Computational_Physics/HW4/HW4_Q1/brusselator.png", dpi=130)
plt.close()
