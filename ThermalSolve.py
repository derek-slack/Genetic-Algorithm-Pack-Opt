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
                     'pack_xyz': [0.2, 0.2], '2d': True,'mesh_xy':[1*40,1*80], 'Buffer': R*0.5}

    parameter_list = []

    for i in range(4):
        for j in range(8):
            np.random.seed(897398*i*j)
            x_1 = Parameter('x_1', float,np.random.rand() ,16 , 'L', R + pack_geometry['Buffer'])
            y_1 = Parameter('y_1', float, np.random.rand(), 16, 'W', R +  pack_geometry['Buffer'])

            L = Parameter('L', float, (np.random.rand())*(1-pack_geometry['cell_n_xyz'][0]+1)*2*(R+pack_geometry['Buffer']) +(pack_geometry['cell_n_xyz'][0]+1)*2*(R+pack_geometry['Buffer']) , 16, 1, (pack_geometry['cell_n_xyz'][0]+1)*2*(R+pack_geometry['Buffer']) )
            W = Parameter('W', float, (np.random.rand())*(1-pack_geometry['cell_n_xyz'][1]+1)*2*(R+pack_geometry['Buffer']) + (pack_geometry['cell_n_xyz'][1]+1)*2*(R+pack_geometry['Buffer']) , 16, 1, (pack_geometry['cell_n_xyz'][1]+1)*2*(R+pack_geometry['Buffer']) )

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

    def apply_constant_temp_bc(T, T_boundary=298.15):
        """Sets edges to a fixed temperature value"""
        T = T.at[0, :].set(T_boundary)  # Left
        T = T.at[-1, :].set(T_boundary)  # Right
        T = T.at[:, 0].set(T_boundary)  # Bottom
        T = T.at[:, -1].set(T_boundary)  # Top
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
            # T_current already has T_amb set at indices 0 and -1
            Txx = (T_current[2:, 1:-1] - 2 * T_current[1:-1, 1:-1] + T_current[:-2, 1:-1]) / dx ** 2
            Tyy = (T_current[1:-1, 2:] - 2 * T_current[1:-1, 1:-1] + T_current[1:-1, :-2]) / dy ** 2
            return alpha * (Txx + Tyy) + q_source

        # Apply Constant Temp BC
        T = apply_constant_temp_bc(T)

        k1 = compute_dT_dt(T)
        T_temp = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * k1)

        # Re-apply for intermediate step
        T_temp = apply_constant_temp_bc(T_temp)

        k2 = compute_dT_dt(T_temp)
        T_new = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * (k1 + k2) / 2)

        # Final clamp
        T_new = apply_constant_temp_bc(T_new)
        return T_new, None


    obj_vmap = jax.vmap(obj_func, in_axes=(0,0,None))

    GA = GeneticAlgorithm(population_size, n_generations, {})

    # best_params = [0.03276011, 0.00280531, 0.,         0.108,      0.18]
    # best_params = [0.02363451 ,0.0193012 , 0.      ,   0.108     , 0.18]
    # best_params = [0.04476722 ,0.03270054, 3.   ,      0.162   ,   0.27]
    best_params = [0.04603575, 0.02522055, 1.     ,    0.16256263 ,0.27083543]
    # L = (pack_geometry['cell_n_xyz'][0]+0.5)*2*(R+pack_geometry['Buffer'])
    # W = (pack_geometry['cell_n_xyz'][1]+0.5)*2*(R+pack_geometry['Buffer'])
    # #
    # L_max = L / 2 - (
    #         pack_geometry['cell_n_xyz'][0] - 1) * (
    #         pack_geometry['cell_rz'][0] + pack_geometry['Buffer'])
    #
    # W_max = W / 2 - (
    #         pack_geometry['cell_n_xyz'][1] - 1) * (
    #         pack_geometry['cell_rz'][0] + pack_geometry['Buffer'])
    #
    # L_min_P = (pack_geometry['cell_n_xyz'][0]+1)*2*(R+pack_geometry['Buffer'])
    # W_min_P = (pack_geometry['cell_n_xyz'][1] + 1) *2* (R+pack_geometry['Buffer'])
    #
    # L_min = R+pack_geometry['Buffer']
    # W_min = R+pack_geometry['Buffer']
    #
    # best_params = [L_max,W_max,2., L_min_P, W_min_P]
    # best_params = [0.00792368, 0.00490037, 2.,         0.05447636, 0.09469337]
    # best_params = [0.01132579, 0.00377359 ,2. ,        0.10816333 ,0.18041291]
    GA.obj_func = obj_vmap
    GA.heat_step_explicit = heat_step_rk2

    GA.T_i = 298
    GA.t_sim = 90*60
    GA.pack_geometry = pack_geometry
    GA.cells  = cells
    GA.T_runaway = 350

    GA.create_population(parameter_list)
    GA.plot_results(best_params)
    final_population = GA.simulate()
    x = np.zeros(population_size)
    y = np.zeros(population_size)

    params_best = GA.best_params()
    print(params_best)
    # [0.04102651 0.13016621 2.         0.09012497 0.1802002]
    # [0.02265409 0.09269854 3.         0.09001389 0.1814139]

    h=1