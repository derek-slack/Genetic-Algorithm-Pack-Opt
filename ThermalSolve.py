from GeneticAlgorithim import GeneticAlgorithm
from Parameter import Parameter
from Chromosome import Chromosome
import numpy as np
import jax
import jax.numpy as jnp
# import matplotlib
# matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from Thermal_Solver import ThermalSolver





if __name__ == '__main__':
    population_size = 32
    n_generations = 50

    # Thermal Init
    class Cell():
        def __init__(self):
            pass
    A = Cell()
    B = Cell()
    C = Cell()
    D = Cell()
    E = Cell()
    F = Cell()
    G = Cell()
    H = Cell()

    cells = [[A,B, C, D],[E, F, G, H]]

    # bounds = (L-2R)/(n-1)
    dz = 0.001
    R = 0.018

    L = 0.2
    Z = dz

    nx = 2
    ny = 2
    nz = 1

    dt_thermal = 1
    #
    pack_geometry = {'cell_rz': [R, Z], 'cell_to_wall_xyz': None, 'cell_n_xyz': [2, 4],
                     'pack_xyz': [0.2, 0.2], '2d': True,'mesh_xy':[40,80], 'Buffer': R*0.3}

    parameter_list = []

    for i in range(4):
        for j in range(8):
            x_1 = Parameter('x_1', float, 1/3,16 , 'L', R + pack_geometry['Buffer'])
            y_1 = Parameter('y_1', float, 1/5, 16, 'W', R +  pack_geometry['Buffer'])

            L = Parameter('L', float, (np.random.rand())*(1-pack_geometry['cell_n_xyz'][0]+1)*R +(pack_geometry['cell_n_xyz'][0]+1)*R, 16, 1, (pack_geometry['cell_n_xyz'][0]+1)*R)
            W = Parameter('W', float, (np.random.rand())*(1-pack_geometry['cell_n_xyz'][1]+1)*R + (pack_geometry['cell_n_xyz'][1]+1)*R, 16, 1, (pack_geometry['cell_n_xyz'][1]+1)*R)

            Pack_Material = Parameter('Pack Material',int,i, 2, 3, 0)

            parameters = {'x': x_1, 'y': y_1,'Pack Material': Pack_Material, 'L':L,'W':W}
            parameter_list.append(parameters)


    def obj_func(params,T_final, max_temp_allowed):
        volume = params[3]*params[4]
        # 1. Calculate the standard penalty
        temp_penalty = 1000 * jnp.maximum(0, (T_final - max_temp_allowed)) ** 2

        # 2. Check for physics explosions (NaN or Infinite values)
        is_invalid = jnp.isnan(T_final) | jnp.isinf(T_final)

        # 3. Apply the Death Penalty
        # If invalid, assign a massive number (1e9). If valid, use the real penalty.
        final_temp_penalty = jnp.where(is_invalid, 1e9, temp_penalty)

        # Continue with your volume penalty addition using final_temp_penalty...
        penalty = 1000 * final_temp_penalty

        return volume + penalty


    def apply_zero_flux_bc(T):
        """Ghost cell method for dT/dn = 0"""
        T = T.at[0, :].set(T[1, :])  # Left: T[-1] = T[0]
        T = T.at[-1, :].set(T[-2, :])  # Right: T[N] = T[N-1]
        T = T.at[:, 0].set(T[:, 1])  # Bottom
        T = T.at[:, -1].set(T[:, -2])  # Top
        return T

    def heat_step_explicit(T,pack_params,Q_dot):
        T = apply_zero_flux_bc(T)

        dx = pack_params[0]
        dy = pack_params[1]
        dt = pack_params[5]

        k = pack_params[2]
        rho = pack_params[3]
        c = pack_params[4]

        Txx = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2
        Tyy = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2

        dT_dt = (k/(rho*c)) * (Txx + Tyy) + Q_dot[1:-1, 1:-1]/ (rho * c)
        T = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * dT_dt)
        return (T,T)


    @jax.jit
    def heat_step_rk2(T, pack_params, Q_dot):
        """RK2 (Heun's method) for better accuracy per timestep"""
        dx = pack_params[0]
        dy = pack_params[1]
        dt = pack_params[5]
        k = pack_params[2]
        rho = pack_params[3]
        c = pack_params[4]

        alpha = k / (rho * c)
        q_source = Q_dot[1:-1, 1:-1] / (rho * c)

        def compute_dT_dt(T_current):
            """Compute time derivative with zero-flux BC built-in"""
            # Zero-flux BC: ghost cells = interior neighbors
            # Avoids separate BC function call
            Txx = (T_current[2:, 1:-1] - 2 * T_current[1:-1, 1:-1] + T_current[:-2, 1:-1]) / dx ** 2
            Tyy = (T_current[1:-1, 2:] - 2 * T_current[1:-1, 1:-1] + T_current[1:-1, :-2]) / dy ** 2
            return alpha * (Txx + Tyy) + q_source

        # Apply BC once at start
        T = apply_zero_flux_bc(T)

        # RK2 stage 1
        k1 = compute_dT_dt(T)
        T_temp = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * k1)

        # Apply BC for intermediate step
        T_temp = apply_zero_flux_bc(T_temp)

        # RK2 stage 2
        k2 = compute_dT_dt(T_temp)

        # Final update (average of slopes)
        T_new = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * (k1 + k2) / 2)
        T_new = apply_zero_flux_bc(T_new)

        return T_new, None


    obj_vmap = jax.vmap(obj_func, in_axes=(0,0,None))

    GA = GeneticAlgorithm(population_size, n_generations, {})
    # best_params = [0.04102651, 0.13016621, 2. ,        0.09012497, 0.1802002]
    # best_params = [0.02265409, 0.09269854/4, 3.,         0.09001389, 0.1814139]

    GA.obj_func = obj_vmap
    GA.heat_step_explicit = heat_step_adi

    GA.T_i = 298
    GA.t_sim = 60*60
    GA.pack_geometry = pack_geometry
    GA.cells  = cells
    GA.T_runaway = 350

    GA.create_population(parameter_list)
    # GA.plot_results(best_params)
    final_population = GA.simulate()
    x = np.zeros(population_size)
    y = np.zeros(population_size)

    params_best = GA.best_params()
    print(params_best)
    # [0.04102651 0.13016621 2.         0.09012497 0.1802002]
    # [0.02265409 0.09269854 3.         0.09001389 0.1814139]

    h=1