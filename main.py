from GeneticAlgorithim import GeneticAlgorithm
from Parameter import Parameter
from Chromosome import Chromosome
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# import matplotlib
# matplotlib.use('Tkagg')
# import matplotlib.pyplot as plt
# from Thermal_Solver import ThermalSolver


# # Thermal Init
#         thermal_solver = ThermalSolver(self.cells,{'c':1000, 'k':50, 'rho':2500}, pack_geometry_dict)
#         thermal_solver.create_mesh(0.001)
#         thermal_solver._place_cell_in_pack()
#         thermal_solver.cell_to_mesh()
#         thermal_solver.plot()
#         self.thermal_solver = thermal_solver
#         self.T_i = np.ones(np.shape(thermal_solver.mesh.active))*300
#         self.thermal_solver.dt = dt_thermal
#         self.thermal_solver.setup_crank_nicolson_matrices()
#
#         self.T_max = T_max

if __name__ == '__main__':
    population_size = 100
    n_generations = 200

    R = 1
    # Thermal Init
    class Cell():
        def __init__(self):
            pass
    A = Cell()
    B = Cell()
    C = Cell()
    D = Cell()

    cells = [[A,B],[C,D]]

    # bounds = (L-2R)/(n-1)
    dz = 0.001
    R = 0.018
    # Z = 0.065
    L = 0.2
    Z = dz

    nx = 2
    ny = 2
    nz = 1

    pack_geometry = {'cell_rz': [R, Z], 'cell_to_wall_xyz': 'even', 'cell_n_xyz': [2, 2], 'pack_xyz': [0.2, 0.2],
                     '2d': True}
    #
    # thermal_solver = ThermalSolver(cells,{'c':1000, 'k':50, 'rho':2500}, pack_geometry)
    # thermal_solver.create_mesh(0.001)
    # thermal_solver._place_cell_in_pack()
    # thermal_solver.cell_to_mesh()
    # thermal_solver.plot()
    #
    # T_i = np.ones(np.shape(thermal_solver.mesh.active))*300
    # thermal_solver.dt = dt_thermal
    #
    # thermal_solver.setup_crank_nicolson_matrices()

    parameter_list = []

    for i in range(10):
        for j in range(10):
            x = Parameter('x', float, (np.random.rand()-1)*10 ,16 , 10, -10)
            y = Parameter('y', float, (np.random.rand()-1)*10, 16, 10, -10)

            parameters = {'x': x, 'y': y}
            parameter_list.append(parameters)

    A = 10
    n = 2

    def obj_func(params_all):
        return A*n + (params_all[0]**2)-(A*jnp.cos(2*jnp.pi*params_all[0])) + (params_all[1]**2)-(A*jnp.cos(2*jnp.pi*params_all[1]))

    obj_vmap = jax.vmap(obj_func)
    GA = GeneticAlgorithm(population_size, n_generations, {})

    GA.create_population(parameter_list)
    final_population = GA.simulate()
    x = np.zeros(population_size)
    y = np.zeros(population_size)

    f = np.zeros(population_size)
    for i, C in enumerate(final_population):
        x[i] = C.parameters['x'].value
        y[i] = C.parameters['y'].value

        f[i] = C.fitness

    params_best = GA.best_params()
    print(params_best)

    h=1