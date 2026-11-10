import numpy as np
import matplotlib.pyplot as plt
from numba import njit


G = 1.0
M = 1.0
r_s = 1e-7         

G_SI, c_SI, pc, Msun, Myr = 6.674e-11, 2.998e8, 3.0857e16, 1.989e30, 3.156e13
L_unit = 100*pc
M_phys = (r_s*L_unit)*c_SI**2/(2*G_SI)                   # kg
t_unit_Myr = np.sqrt(L_unit**3/(G_SI*M_phys))/Myr        # one code time unit in Myr
print(f"M = {M_phys/Msun:.3e} Msun,  1 time unit = {t_unit_Myr:.3f} Myr")

vy0 = 0.8*np.sqrt(G*M/4)        # 0.8 of circular speed (force law is GM/(4 r^2))
r0 = np.array([1.0, 0.0, 0.0, vy0])   # [x, vx, y, vy]
d = 1e-7                        # tolerance per unit time (from part a)
h0 = 0.1

@njit
def f(r, A, B):
    x, vx, y, vy = r[0], r[1], r[2], r[3]
    rbh = np.sqrt(x*x + y*y)
    vbh = np.sqrt(vx*vx + vy*vy)
    drag = -A/(vbh**3 + B)
    out = np.empty(4)
    out[0] = vx
    out[1] = -G*M*x/(4*rbh**3) + drag*vx
    out[2] = vy
    out[3] = -G*M*y/(4*rbh**3) + drag*vy
    return out

@njit
def RK4(r, step, A, B):
    k1 = step*f(r, A, B)
    k2 = step*f(r + 0.5*k1, A, B)
    k3 = step*f(r + 0.5*k2, A, B)
    k4 = step*f(r + k3, A, B)
    return (k1 + 2*k2 + 2*k3 + k4)/6

@njit
def time_to_rs(r0, A, B, d, h0, rstop, max_steps=10_000_000):
    r = r0.copy()
    t = 0.0
    h = h0
    for _ in range(max_steps):
        if np.sqrt(r[0]**2 + r[2]**2) < rstop:
            return t
        d1 = RK4(r, h, A, B)
        d2 = RK4(r + d1, h, A, B)
        dd = d1 + d2
        r2h = RK4(r, 2*h, A, B)
        err = np.sqrt((dd[0]-r2h[0])**2 + (dd[2]-r2h[2])**2)
        rho = h*d/err if err > 0 else 16.0
        if rho >= 1:
            r += dd
            t += 2*h
            h *= min(rho**0.25, 2.0)
        else:
            h *= rho**0.25
    return -1.0

Avals = np.logspace(np.log10(0.5), 1, 6)
Bvals = np.logspace(np.log10(0.5), 1, 6)

ratios, times_Myr, Acolor = [], [], []
for A in Avals:
    for B in Bvals:
        t_end = time_to_rs(r0, A, B, d, h0, r_s)
        if t_end < 0:
            print(f"A={A:.2f}, B={B:.2f}: did not reach r_s")
            continue
        ratios.append(B/A)
        times_Myr.append(t_end*t_unit_Myr)
        Acolor.append(A)

ratios = np.array(ratios); times_Myr = np.array(times_Myr); Acolor = np.array(Acolor)

fig, ax = plt.subplots(figsize=(7, 4.5))
sc = ax.scatter(ratios, times_Myr, c=Acolor, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("B/A")
ax.set_ylabel("Time to reach $r_s$ [Myr]")
ax.set_title("Inspiral time vs. B/A")
fig.colorbar(sc, label="A")
plt.tight_layout()
plt.savefig("timetoreachrS.png", dpi=200)
plt.show()