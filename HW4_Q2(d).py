import numpy as np
import matplotlib.pyplot as plt
from numba import njit

G, M = 1.0, 1.0
r_s = 1e-7
d = 1e-7
h0 = 0.1

G_SI, c_SI, pc, Myr = 6.674e-11, 2.998e8, 3.0857e16, 3.156e13
L_unit = 100*pc
M_phys = (r_s*L_unit)*c_SI**2/(2*G_SI)
t_unit_Myr = np.sqrt(L_unit**3/(G_SI*M_phys))/Myr

vcirc = np.sqrt(G*M/4)

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

def T(A, B, vfrac):
    r0 = np.array([1.0, 0.0, 0.0, vfrac*vcirc])
    t = time_to_rs(r0, A, B, d, h0, r_s)
    return t*t_unit_Myr if t >= 0 else np.nan


vfracs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ratios = np.logspace(np.log10(0.05), np.log10(20), 12)

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
for vf in vfracs:
    ts = [T(1.0, rr, vf) for rr in ratios]
    ax1.plot(ratios, ts, 'o-', label=f"v0 = {vf} v_circ")
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel("B/A  (A = 1)"); ax1.set_ylabel("Time to reach $r_s$ [Myr]")
ax1.set_title("Dependence on initial velocity")
ax1.legend(fontsize=8)
plt.tight_layout(); plt.savefig("d_velocity.png", dpi=200)

Avals = [0.5, 1, 2, 5, 10]
ratios2 = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])

fig2, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.5))
for A in Avals:
    ts = np.array([T(A, rr*A, 0.8) for rr in ratios2])
    axa.plot(ratios2, ts, 'o-', label=f"A = {A}")
    axb.plot(ratios2, ts*A/ratios2/ (t_unit_Myr/t_unit_Myr), 'o-', label=f"A = {A}")  # t*A/(B/A)... see below
axa.set_xscale('log'); axa.set_yscale('log')
axa.set_xlabel("B/A"); axa.set_ylabel("Time [Myr]")
axa.set_title("Same B/A, different A"); axa.legend()

axb.set_xscale('log'); axb.set_yscale('log')
axb.set_xlabel("B/A"); axb.set_ylabel(r"$t\,A$ / (B/A)  [arb.]")
axb.set_title("Test of scaling t ~ f(B/A)/A"); axb.legend()
plt.tight_layout(); plt.savefig("d_ratio_vs_individual.png", dpi=200)


print("B/A = 1, v0 = 0.8 v_circ")
for A in Avals:
    print(f"  A = B = {A:5.1f}:  t = {T(A, A, 0.8):.4g} Myr")
print("\nB/A = 1, A = 1, varying v0")
for vf in vfracs:
    print(f"  v0 = {vf:4.2f} v_circ:  t = {T(1.0, 1.0, vf):.4g} Myr")

plt.show()