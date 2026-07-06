"""
central_force_orbit.py
=======================

Simulates the trajectory ("orbit") of a particle moving under an attractive
central force of the general power-law form:

    F(r) = -k * r**n            (n can be ANY real number, positive or negative)

    n = -2  -> inverse-square law (gravity / Coulomb)   -> closed ellipse (Kepler)
    n = -1  -> logarithmic potential                     -> spiral / precessing orbit
    n =  1  -> Hooke's law (linear restoring force)       -> closed ellipse (centered at origin)
    n = -3  -> classic unstable orbit example (collapses into the center)
    other n -> generally an OPEN, precessing "rosette" orbit (Bertrand's theorem:
               only n = 1 and n = -2 give orbits that always close)

The equations of motion are integrated directly in Cartesian coordinates:

    m * d2x/dt2 = F(r) * (x/r)
    m * d2y/dt2 = F(r) * (y/r)

so the code works for ANY n without needing a special-case solution.

Usage
-----
Just edit the parameters in the `if __name__ == "__main__":` block at the
bottom (mass, k, n, initial position/velocity, simulation time) and run:

    python central_force_orbit.py

This will produce:
  1. A static plot of the orbit (x vs y) with the force center marked.
  2. A plot of r(t) - the radial distance vs time.
  3. A plot showing conservation of energy and angular momentum (sanity check).
  4. An optional animation of the particle moving along its orbit.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------

def central_force_magnitude(r, k, n):
    """
    Radial force magnitude for F(r) = -k * r**n.
    Returns the SIGNED radial force (negative => attractive, pulling inward).

    Note: r is always > 0 (it's a distance), so r**n is well defined for any
    real n as long as r != 0.
    """
    return -k * r**n


def equations_of_motion(t, state, k, n, m):
    """
    state = [x, y, vx, vy]
    Returns d(state)/dt = [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    r = np.hypot(x, y)

    if r < 1e-9:
        # Avoid division by zero / singular blow-up right at the center.
        r = 1e-9

    F = central_force_magnitude(r, k, n)   # signed magnitude along r_hat
    ax = (F / m) * (x / r)
    ay = (F / m) * (y / r)

    return [vx, vy, ax, ay]


def potential_energy(r, k, n):
    """
    Potential energy U(r) such that F(r) = -dU/dr = -k*r^n
      => U(r) = k * r**(n+1) / (n+1),  for n != -1
      => U(r) = -k * ln(r),            for n == -1
    (integration constant chosen as 0)
    """
    if np.isclose(n, -1.0):
        return -k * np.log(r)
    return k * r**(n + 1) / (n + 1)


def angular_momentum(x, y, vx, vy, m):
    """L_z = m * (x*vy - y*vx), conserved for any central force."""
    return m * (x * vy - y * vx)


def total_energy(x, y, vx, vy, m, k, n):
    r = np.hypot(x, y)
    KE = 0.5 * m * (vx**2 + vy**2)
    PE = potential_energy(r, k, n)
    return KE + PE


def find_apsides(r):
    """
    Locate periapsis (r_min) and apoapsis (r_max) points from the simulated
    r(t) time series by finding its local minima/maxima. Returns the mean
    r_min and r_max over all detected apsides (more robust than a single
    passage), plus the raw index arrays in case you want to inspect them.
    """
    min_idx = argrelextrema(r, np.less_equal, order=3)[0]
    max_idx = argrelextrema(r, np.greater_equal, order=3)[0]

    # argrelextrema with *_equal can return duplicate/adjacent points on flat
    # regions; collapse consecutive runs down to a single representative index.
    def _dedupe(idx_array):
        if len(idx_array) == 0:
            return idx_array
        groups = np.split(idx_array, np.where(np.diff(idx_array) > 1)[0] + 1)
        return np.array([g[len(g) // 2] for g in groups])

    min_idx = _dedupe(min_idx)
    max_idx = _dedupe(max_idx)

    r_min = np.mean(r[min_idx]) if len(min_idx) else r.min()
    r_max = np.mean(r[max_idx]) if len(max_idx) else r.max()

    return r_min, r_max, min_idx, max_idx


def orbit_eccentricity(result, m, k, n):
    """
    Estimate orbital eccentricity from the simulated trajectory.

    Returns a dict containing:
      - r_min, r_max          : periapsis / apoapsis distances (numerical)
      - e_focus_formula       : e = (r_max - r_min)/(r_max + r_min)
                                 [CORRECT interpretation only when the force
                                  center sits at a focus of the orbit, i.e. n = -2]
      - e_centered_formula    : e = sqrt(1 - (r_min/r_max)^2)
                                 [CORRECT interpretation only when the force
                                  center sits at the CENTER of the ellipse, i.e. n = 1]
      - e_exact_kepler        : exact analytic e from E, L (only computed if n = -2)
      - e_exact_harmonic      : exact analytic e from E, L (only computed if n = 1)

    For any other n, treat e_focus_formula / e_centered_formula only as rough
    "how elongated is this orbit" indicators -- the orbit is generally an open,
    precessing rosette, not a closed conic, so no single eccentricity strictly applies.
    """
    r = result["r"]
    r_min, r_max, min_idx, max_idx = find_apsides(r)

    out = {"r_min": r_min, "r_max": r_max}

    if r_max > 0:
        out["e_focus_formula"] = (r_max - r_min) / (r_max + r_min)
        ratio = r_min / r_max
        out["e_centered_formula"] = np.sqrt(max(0.0, 1 - ratio**2))
    else:
        out["e_focus_formula"] = np.nan
        out["e_centered_formula"] = np.nan

    E0 = result["E"][0]
    L0 = result["L"][0]

    if np.isclose(n, -2.0):
        # U(r) = -k/r  =>  e = sqrt(1 + 2*E*L^2 / (m*k^2))
        out["e_exact_kepler"] = np.sqrt(max(0.0, 1 + 2 * E0 * L0**2 / (m * k**2)))

    if np.isclose(n, 1.0):
        # Isotropic harmonic oscillator: U(r) = 1/2 * k * r^2, omega^2 = k/m
        # Orbit is an ellipse centered at the origin with semi-axes a >= b:
        #   a^2 + b^2 = 2E / (m*omega^2),   a*b = L / (m*omega)
        omega = np.sqrt(k / m)
        s = 2 * E0 / (m * omega**2)        # a^2 + b^2
        p = abs(L0) / (m * omega)          # a*b
        disc = max(s**2 - 4 * p**2, 0.0)
        a2 = (s + np.sqrt(disc)) / 2
        b2 = (s - np.sqrt(disc)) / 2
        out["semi_major_exact"] = np.sqrt(a2)
        out["semi_minor_exact"] = np.sqrt(b2)
        out["e_exact_harmonic"] = np.sqrt(max(0.0, 1 - b2 / a2)) if a2 > 0 else np.nan

    return out


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

def simulate(m, k, n, x0, y0, vx0, vy0, t_max, n_points=4000):
    """
    Integrate the orbit and return a dict with time, position, velocity,
    energy and angular momentum arrays.
    """
    state0 = [x0, y0, vx0, vy0]
    t_eval = np.linspace(0, t_max, n_points)

    sol = solve_ivp(
        equations_of_motion,
        t_span=(0, t_max),
        y0=state0,
        args=(k, n, m),
        t_eval=t_eval,
        method="DOP853",       # high-accuracy explicit RK, good for orbital problems
        rtol=1e-9,
        atol=1e-9,
        max_step=t_max / 2000,
    )

    x, y, vx, vy = sol.y
    r = np.hypot(x, y)
    E = total_energy(x, y, vx, vy, m, k, n)
    L = angular_momentum(x, y, vx, vy, m)

    return {
        "t": sol.t,
        "x": x, "y": y, "vx": vx, "vy": vy,
        "r": r, "E": E, "L": L,
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_orbit(result, n, k, ecc=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(result["x"], result["y"], lw=1.2, color="steelblue", label="orbit path")
    ax.plot(0, 0, "o", color="crimson", markersize=10, label="force center")
    ax.plot(result["x"][0], result["y"][0], "go", markersize=7, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Orbit under F(r) = -k r^n   (n = {n:g}, k = {k:g})")

    if ecc is not None:
        info = (
            f"r_min = {ecc['r_min']:.4g}\n"
            f"r_max = {ecc['r_max']:.4g}\n"
        )
        if np.isclose(n, -2.0):
            info += f"e (Kepler, exact) = {ecc['e_exact_kepler']:.4f}"
        elif np.isclose(n, 1.0):
            info += f"e (harmonic, exact) = {ecc['e_exact_harmonic']:.4f}"
        else:
            info += (
                f"e (focus formula) = {ecc['e_focus_formula']:.4f}\n"
                f"e (centered formula) = {ecc['e_centered_formula']:.4f}\n"
                f"(orbit generally not a closed conic\n for this n -- see console notes)"
            )
        ax.text(
            0.02, 0.02, info, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_diagnostics(result):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(result["t"], result["r"], color="darkorange")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("r(t)")
    axes[0].set_title("Radial distance vs time")
    axes[0].grid(alpha=0.3)

    # Normalize energy/L to their initial values to show conservation clearly
    E0 = result["E"][0]
    L0 = result["L"][0]

    axes[1].plot(result["t"], (result["E"] - E0) / max(abs(E0), 1e-12))
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("relative energy drift")
    axes[1].set_title("Energy conservation check")
    axes[1].grid(alpha=0.3)

    axes[2].plot(result["t"], (result["L"] - L0) / max(abs(L0), 1e-12))
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("relative L drift")
    axes[2].set_title("Angular momentum conservation check")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def animate_orbit(result, n, k, save_path=None, fps=30):
    """
    Animate the particle moving along its orbit. If save_path is given
    (e.g. 'orbit.mp4' or 'orbit.gif'), the animation is saved to disk
    instead of (or in addition to) being shown.
    """
    x, y = result["x"], result["y"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, lw=0.8, color="lightsteelblue")
    ax.plot(0, 0, "o", color="crimson", markersize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Orbit animation  (n = {n:g}, k = {k:g})")
    ax.grid(alpha=0.3)

    trail, = ax.plot([], [], lw=2, color="steelblue")
    point, = ax.plot([], [], "o", color="darkgreen", markersize=8)

    trail_len = max(len(x) // 20, 20)  # how much recent trail to keep highlighted

    def init():
        trail.set_data([], [])
        point.set_data([], [])
        return trail, point

    def update(frame):
        start = max(0, frame - trail_len)
        trail.set_data(x[start:frame + 1], y[start:frame + 1])
        point.set_data([x[frame]], [y[frame]])
        return trail, point

    step = max(1, len(x) // 600)  # subsample frames for a smoother/faster animation
    frames = range(0, len(x), step)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=1000 / fps, blit=True,
    )

    if save_path:
        ani.save(save_path, fps=fps)
        print(f"Animation saved to {save_path}")

    return ani


# ----------------------------------------------------------------------
# Main / example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Parameters you can freely change --------------------------------
    m = 1.0          # particle mass
    k = 1.0          # force strength, F(r) = -k * r^n
    n = -2.0         # <-- THE EXPONENT: any real number. Try -2, -1, 1, -2.5, 0.5 ...

    # Initial conditions
    x0, y0 = 1.0, 0.0
    vx0, vy0 = 0.0, 1.0

    t_max = 20.0     # total simulation time
    # ------------------------------------------------------------------

    result = simulate(m, k, n, x0, y0, vx0, vy0, t_max)

    ecc = orbit_eccentricity(result, m, k, n)
    print("\n--- Orbit shape ---")
    print(f"r_min (periapsis) = {ecc['r_min']:.6g}")
    print(f"r_max (apoapsis)  = {ecc['r_max']:.6g}")
    if np.isclose(n, -2.0):
        print(f"Eccentricity (exact, Kepler formula)   = {ecc['e_exact_kepler']:.6f}")
    elif np.isclose(n, 1.0):
        print(f"Eccentricity (exact, harmonic formula) = {ecc['e_exact_harmonic']:.6f}")
        print(f"Semi-major axis a = {ecc['semi_major_exact']:.6f}, "
              f"semi-minor axis b = {ecc['semi_minor_exact']:.6f}")
    else:
        print(f"e, focus-ellipse formula (r_max-r_min)/(r_max+r_min) = {ecc['e_focus_formula']:.6f}")
        print(f"e, centered-ellipse formula sqrt(1-(r_min/r_max)^2)  = {ecc['e_centered_formula']:.6f}")
        print("Note: for n != -2 and n != 1 the orbit is generally an open,")
        print("precessing rosette -- these two numbers are just shape estimates,")
        print("not a rigorous 'the' eccentricity (no single conic describes the path).")

    fig1 = plot_orbit(result, n, k, ecc=ecc)
    fig2 = plot_diagnostics(result)

    # Uncomment to also generate an animation window:
    # ani = animate_orbit(result, n, k)

    # Uncomment to save the animation to a file instead of/as well as showing plots:
    # ani = animate_orbit(result, n, k, save_path="orbit.gif", fps=30)

    plt.show()
