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
    n_generations = 100

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
                     'pack_xyz': [0.2, 0.2], '2d': True,'mesh_xy':[200,450], 'Buffer': R*0.25}

    parameter_list = []

    for i in range(4):
        for j in range(8):
            np.random.seed(897398*i*j)
            x_1 = Parameter('x_1', float,np.random.rand() ,16 , 'L', R + pack_geometry['Buffer'])
            np.random.seed(344321 * i * j)
            y_1 = Parameter('y_1', float, np.random.rand(), 16, 'W', R +  pack_geometry['Buffer'])
            np.random.seed(8234398 * i * j)
            L = Parameter('L', float, (np.random.rand())*(pack_geometry['cell_n_xyz'][0])*2*(R) +(pack_geometry['cell_n_xyz'][0]+1)*2*(R+pack_geometry['Buffer']) , 16, 0.5, (pack_geometry['cell_n_xyz'][0]+1)*2*(R) )
            np.random.seed(35734 * i * j)
            W = Parameter('W', float, (np.random.rand())*(pack_geometry['cell_n_xyz'][1])*2*(R) + (pack_geometry['cell_n_xyz'][1]+1)*2*(R+pack_geometry['Buffer']) , 16,0.5, (pack_geometry['cell_n_xyz'][1]+1)*2*(R) )

            Pack_Material = Parameter('Pack Material',int,i, 2, 3, 0)

            parameters = {'x': x_1, 'y': y_1,'Pack Material': Pack_Material, 'L':L,'W':W}
            parameter_list.append(parameters)


    def obj_func(params,T_final,rho, max_temp_allowed, geometric_error):
        weight = rho*params[3]*params[4]

        temp_penalty = 1000 * jnp.maximum(0, (T_final - max_temp_allowed)) ** 2

        is_invalid = jnp.isnan(T_final) | jnp.isinf(T_final)

        final_temp_penalty = jnp.where(is_invalid, 1e9, temp_penalty)
        geo_penalty = jnp.where(abs(geometric_error)>0.075, 1e9, 0)


        return weight + final_temp_penalty + geo_penalty


    def apply_convection_bc(T,k,h,dx,T_boundary=298.15):

        T = T.at[0, :].set((k*T[1, :] + h*dx*T_boundary)/(k+h*dx))  # Top
        T = T.at[-1, :].set((k*T[-2, :] + h*dx*T_boundary)/(k+h*dx))  # Bottom
        # T = T.at[:, 0].set((k*T[:, 1] + h*dx*T_boundary)/(k+h*dx))   # Left
        T = T.at[:, 0].set(T[:, 1])
        # T = T.at[:, -1].set((k*T[:, -2] + h*dx*T_boundary)/(k+h*dx)) # Right
        T = T.at[:, -1].set(T[:, -2])
        return T

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
            Txx = (T_current[2:, 1:-1] - 2 * T_current[1:-1, 1:-1] + T_current[:-2, 1:-1]) / dx ** 2
            Tyy = (T_current[1:-1, 2:] - 2 * T_current[1:-1, 1:-1] + T_current[1:-1, :-2]) / dy ** 2
            return alpha * (Txx + Tyy) + q_source

        h = 10
        T = apply_convection_bc(T,k,h,dx)

        k1 = compute_dT_dt(T)
        T_temp = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * k1)

        T_temp = apply_convection_bc(T_temp,k,h,dx)

        k2 = compute_dT_dt(T_temp)
        T_new = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * (k1 + k2) / 2)

        T_new = apply_convection_bc(T_new,k,h,dx)

        return T_new, None


    obj_vmap = jax.vmap(obj_func, in_axes=(0,0,0,None,0))

    GA = GeneticAlgorithm(population_size, n_generations, {}, thermal_solve=True)

    best_params = [0.03024601, 0.04079956, 3.  ,       0.12281627 ,0.28548989]
    GA.obj_func = obj_vmap
    GA.heat_step_explicit = heat_step_rk2

    GA.T_i = 298
    GA.t_sim = 60*60
    GA.pack_geometry = pack_geometry
    GA.cells  = cells
    GA.T_runaway = 350

    GA.create_population(parameter_list)
    # GA.plot_results(best_params)
    T_out, T_all = GA.plot_thermal(best_params)
    # GA.simulate()


    params_best = GA.population
    for p in params_best:
        print(f'{p.read_params()},fit:{p.fitness}')
