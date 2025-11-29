import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    """
    Lattice Boltzmann Simulation (D2Q9) for Lid-Driven Cavity
    with Internally Heated Baffles.

    UPDATED:
    1. Grid increased to 256x256 (High Resolution).
    2. Iterations increased to 20,000 (required for larger grid).
    3. Visualization style: Professional Hybrid (Smooth Color + Thin Contours).
    """

    # =========================================================================
    # 1. Simulation Parameters
    # =========================================================================
    # Geometry (High Resolution)
    nx, ny = 256, 256         # Grid dimensions
    max_iter = 20000          # Increased time steps for larger grid

    # Baffle Geometry (Horizontal plates on side walls)
    baffle_y_loc = ny // 2    # Height of baffles (middle)
    baffle_length = nx // 4   # Length of each baffle
    baffle_thickness = 10     # Thicker baffles for higher res (approx 4% of height)

    # Physical Parameters
    u_lid = 0.1               # Lid velocity (lattice units)
    Re = 100.0                # Reynolds Number
    Pr = 0.71                 # Prandtl Number (Air)

    # Thermal Boundary Conditions
    T_cold = 0.0              # Top wall temperature
    T_hot = 1.0               # Baffle temperature
    T_ref = 0.0               # Initial fluid temperature

    # Lattice Constants (D2Q9)
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])

    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                  1/36, 1/36, 1/36, 1/36])

    noslip = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    # Derived LBM Parameters
    nu = u_lid * nx / Re                # Kinematic viscosity
    alpha = nu / Pr                     # Thermal diffusivity
    omega = 1.0 / (3.0 * nu + 0.5)      # Fluid relaxation parameter
    omega_t = 1.0 / (3.0 * alpha + 0.5) # Thermal relaxation parameter

    print(f"Simulation Setup:")
    print(f"Grid: {nx}x{ny}, Re: {Re}, Pr: {Pr}")
    print(f"Relaxation times -> Fluid: {1/omega:.3f}, Thermal: {1/omega_t:.3f}")

    # =========================================================================
    # 2. Initialization
    # =========================================================================
    # Define Baffle Mask
    obstacle = np.full((nx, ny), False, dtype=bool)

    # Left Baffle
    obstacle[0:baffle_length,
             baffle_y_loc:baffle_y_loc+baffle_thickness] = True
    # Right Baffle
    obstacle[nx-baffle_length:nx,
             baffle_y_loc:baffle_y_loc+baffle_thickness] = True

    # Arrays
    f = np.zeros((nx, ny, 9))
    g = np.zeros((nx, ny, 9))
    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny, 2))
    T = np.full((nx, ny), T_ref)

    # Equilibrium function
    def get_equilibrium(rho_field, u_field, T_field, type='fluid'):
        eq = np.zeros((nx, ny, 9))
        u2 = u_field[..., 0]**2 + u_field[..., 1]**2
        for i in range(9):
            cu = (c[i,0]*u_field[...,0] + c[i,1]*u_field[...,1])
            if type == 'fluid':
                eq[...,i] = rho_field * w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
            elif type == 'thermal':
                eq[...,i] = T_field * w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        return eq

    f = get_equilibrium(rho, u, T, 'fluid')
    g = get_equilibrium(rho, u, T, 'thermal')

    # =========================================================================
    # 3. Main Simulation Loop
    # =========================================================================
    print("Starting simulation...")

    for time_step in range(max_iter):

        # Macroscopic properties
        rho = np.sum(f, axis=2)
        u[..., 0] = (np.sum(f * c[:, 0], axis=2)) / rho
        u[..., 1] = (np.sum(f * c[:, 1], axis=2)) / rho
        T = np.sum(g, axis=2)

        # Force Macroscopic BCs
        u[:, -1, 0] = u_lid
        u[:, -1, 1] = 0
        T[:, -1] = T_cold
        u[obstacle, :] = 0.0
        T[obstacle] = T_hot

        # Collision
        feq = get_equilibrium(rho, u, T, 'fluid')
        geq = get_equilibrium(rho, u, T, 'thermal')
        f = f * (1 - omega) + feq * omega
        g = g * (1 - omega_t) + geq * omega_t

        # Streaming
        for i in range(9):
            f[..., i] = np.roll(f[..., i], shift=c[i], axis=(0, 1))
            g[..., i] = np.roll(g[..., i], shift=c[i], axis=(0, 1))

        # Boundary Conditions (Bounce-back)
        for i in range(9):
            f[obstacle, noslip[i]] = f[obstacle, i] # Obstacles
            if c[i,0] < 0: f[0, :, i] = f[0, :, noslip[i]] # Left
            if c[i,0] > 0: f[-1, :, i] = f[-1, :, noslip[i]] # Right
            if c[i,1] < 0: f[:, 0, i] = f[:, 0, noslip[i]] # Bottom
            if c[i,1] > 0: # Moving Lid
                opp = noslip[i]
                density_term = 6.0 * w[i] * rho[:, -1] * c[i,0] * u_lid
                f[:, -1, opp] = f[:, -1, i] - density_term

        # Thermal BCs
        g[:, -1, :] = get_equilibrium(rho[:, -1], u[:, -1], T_cold, 'thermal')[:, -1, :]
        g_baffle_eq = get_equilibrium(rho, u, T_hot, 'thermal')
        g[obstacle, :] = g_baffle_eq[obstacle, :]
        g[:, 0, :] = g[:, 1, :]   # Bottom adiabatic
        g[0, :, :] = g[1, :, :]   # Left adiabatic
        g[-1, :, :] = g[-2, :, :] # Right adiabatic

        if time_step % 2000 == 0:
            print(f"Step {time_step}/{max_iter} complete...")

    # =========================================================================
    # 4. Visualization (High Res & Professional Style)
    # =========================================================================
    print("Simulation complete. Generating plots...")

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Mask velocity for plotting
    u_plot_x = u[..., 0].copy()
    u_plot_y = u[..., 1].copy()
    u_plot_x[obstacle] = 0.0
    u_plot_y[obstacle] = 0.0
    u_mag = np.sqrt(u_plot_x**2 + u_plot_y**2)
    u_mag[obstacle] = np.nan

    plt.figure(figsize=(14, 6))

    # Plot 1: Velocity Streamlines
    plt.subplot(1, 2, 1)
    plt.title(f"Velocity Streamlines (Re={Re}, Grid={nx}x{ny})")
    # Use interpolation='bicubic' for smoother background
    plt.imshow(u_mag.T, origin='lower', cmap='Blues', alpha=0.6, interpolation='bicubic')
    # Streamlines
    plt.streamplot(X, Y, u_plot_x.T, u_plot_y.T, color='k', density=1.2, linewidth=0.5, arrowsize=0.7)
    # Mask Obstacles
    plt.imshow(obstacle.T, origin='lower', cmap='Greys', alpha=0.6)
    plt.xlim(0, nx-1)
    plt.ylim(0, ny-1)
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot 2: Temperature (Hybrid Style: Smooth Color + Thin Lines)
    plt.subplot(1, 2, 2)
    plt.title(f"Temperature Distribution (Pr={Pr})")

    # Smooth background gradient
    plt.imshow(T.T, origin='lower', cmap='RdBu_r', alpha=0.6, interpolation='bicubic')

    # Thin black contour lines (Less clutter)
    CS = plt.contour(X, Y, T.T, levels=12, colors='k', linewidths=0.6, alpha=0.8)
    plt.clabel(CS, inline=True, fontsize=8, fmt='%1.2f')

    # Solid Baffles (Hiding internal noise)
    plt.imshow(obstacle.T, origin='lower', cmap='Greys', alpha=1.0)

    plt.xlim(0, nx-1)
    plt.ylim(0, ny-1)
    plt.xlabel("x")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
