from GeneticAlgorithim import GeneticAlgorithm
from Parameter import Parameter
from Chromosome import Chromosome
import numpy as np
import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


if __name__ == '__main__':
    population_size = 200
    n_generations = 1000


    parameter_list = []

    for i in range(10):
        for j in range(20):
            x = Parameter('x', float, (np.random.rand())*10-5 ,12 , 5, -5)
            y = Parameter('y', float, (np.random.rand())*10-5, 12, 5, -5)

            parameters = {'x': x, 'y': y}
            parameter_list.append(parameters)

    A = 10
    n = 2

    def obj_func(params_all):
        return A*n + (params_all[0]**2)-(A*jnp.cos(2*jnp.pi*params_all[0])) + (params_all[1]**2)-(A*jnp.cos(2*jnp.pi*params_all[1]))

    GA_params = {'Crossover Rate': 0.6, 'Elitism':2, 'Mutation Rate': 0.15}

    obj_vmap = jax.vmap(obj_func)
    GA = GeneticAlgorithm(population_size, n_generations, GA_params)
    GA.obj_func = obj_func

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