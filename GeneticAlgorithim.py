import copy

import jax
import jax.numpy as jnp
import numpy as np

from Chromosome import Chromosome

from Thermal_Solver import ThermalSolver


class GeneticAlgorithm():
    def __init__(self, population_size, n_generations, parameters):
        self.population_size = population_size
        self.n_generations = n_generations
        self.process_parameters(parameters)

    def process_parameters(self, parameters):
        default_params = {'Fitness':0.6,'Crossover Rate':0.6,'Elitism':4, 'Mutation Rate':0.10}
        self.Fitness_cutoff = default_params['Fitness']
        self.crossover_rate = default_params['Crossover Rate']
        self.elitism = default_params['Elitism']
        self.mutation_rate = default_params['Mutation Rate']

    def create_population(self, initial_params: list[dict]):
        population = []

        for n in range(self.population_size):
            C = Chromosome(initial_params[n],self.pack_geometry)
            C.encode()
            population.append(C)
        self.population = population

    def evaluation_fitness(self, population):
        params_all = []
        pack_params_all = []
        ts_all = []
        Q_dot_all = []
        dt_old = 100
        for C in population:
            p = C.read_params()
            params_all.append(p)
            ts, c, k, rho, dx, dy = self.init_thermal_solve(p)
            dt_new = 0.45 / (k / (rho * c) * (1 / (dx ** 2) + 1 / (dy ** 2)))
            # dt_new = 2
            dt_old = jnp.minimum(dt_new, dt_old)
            pack_params_all.append([dx,dy,k, rho, c])
            ts_all.append(ts)

            thickness_z = 0.008
            active_volume = jnp.sum(ts.mesh.generation_cells) * dx * dy * thickness_z
            volumetric_q = 0.5 / active_volume
            Q_dot_i = ts.mesh.generation_cells * volumetric_q
            Q_dot_all.append(Q_dot_i)
        print(f'dt set to {dt_old}')
        time_step = jnp.arange(0, self.t_sim,step=dt_old)
        Ti_all = jnp.ones(np.shape(ts_all[0].mesh.active)) * 298

        params_all = jnp.array(params_all)
        pack_params_all = jnp.array(pack_params_all)
        pack_params_all = jnp.hstack([pack_params_all, np.ones([len(population),1])*dt_old])
        # Ti_all = jnp.array(Ti_all)
        Q_dot_all = jnp.array(Q_dot_all)
        T = Ti_all
        T_max = self.f(Ti_all, Q_dot_all, pack_params_all, time_step)

        fitness = self.obj_func(params_all, T_max, self.T_runaway)
        print(f'Max temp all: {jnp.max(T_max)}')
        for i, C in enumerate(population):
            C.fitness = fitness[i].item()
    def selection(self):
        selected = np.zeros(self.population_size)
        fitness_sum = 0
        min_fitness = 1e10
        max_fitness = 0
        for i, C in enumerate(self.population):
            min_fitness = min(min_fitness,C.fitness)
            max_fitness = max(max_fitness, C.fitness)
        for i, C in enumerate(self.population):
            fitness_sum += max_fitness - C.fitness
        for i, C in enumerate(self.population):
            selected[i] = (max_fitness - C.fitness)/fitness_sum

        fitness_values = [C.fitness for C in self.population]
        ind_best = np.argsort(fitness_values)
        elite = [self.population[i] for i in ind_best[:self.elitism]]
        self.elite = elite
        population_pairs = self.spin_wheel(selected)
        print(f'min fitness: {min_fitness}')
        print(f'average fitness: {sum(C.fitness for C in self.population)/self.population_size}')
        return population_pairs

    def mutate(self, C):
        for param in C.parameters_encoded.keys():
            encoded = C.parameters_encoded[param].value
            for i in range(len(encoded)):
                if np.random.rand() < self.mutation_rate:
                    encoded = encoded[:i] + str(1 - int(encoded[i])) + encoded[i + 1:]
            C.parameters_encoded[param].value = encoded
    def spin_wheel(self, selected):
        population_pairs_1 = np.random.choice(np.linspace(0,self.population_size-1,self.population_size),int((self.population_size-self.elitism)/2),p=selected)
        population_pairs_2 = np.random.choice(np.linspace(0, self.population_size-1, self.population_size),
                                              int((self.population_size-self.elitism)/2), p=selected)
        return np.array([population_pairs_1,population_pairs_2])

    def crossover(self, population_pairs):
        new_pop = []
        for i in range(len(population_pairs[0])):
            pair = population_pairs[:,i]
            C1 = self.population[int(pair[0])]
            C2 = self.population[int(pair[1])]
            C1_new = copy.deepcopy(C1)
            C2_new = copy.deepcopy(C2)
            self.perform_crossover(C1,C2, C1_new,C2_new)
            new_pop.append(C1_new)
            new_pop.append(C2_new)
        return new_pop
    def perform_crossover(self,C1,C2, C1_new,C2_new):
        for param in C1.parameters.keys():
            if np.random.rand(1) < self.crossover_rate:
                C1_Param = C1.parameters_encoded[param].value
                C2_Param = C2.parameters_encoded[param].value
                crossover_point = np.random.randint(len(C1_Param))
                C1_new_param = C1_Param[0:crossover_point] + C2_Param[crossover_point:]
                C2_new_param = C2_Param[0:crossover_point] + C1_Param[crossover_point:]

                C1_new.parameters_encoded[param].value = C1_new_param
                C2_new.parameters_encoded[param].value = C2_new_param

    def simulate(self):

        self.f = self.prepare_solve()
        self.evaluation_fitness(self.population)

        for i in range(self.n_generations):
            pop_pairs = self.selection()
            new_pop = self.crossover(pop_pairs)
            for C in new_pop:
                self.mutate(C)

            self.evaluation_fitness(new_pop)
            self.population = new_pop + self.elite

        return self.population

    def init_thermal_solve(self,params):
        pack_geometry = copy.deepcopy(self.pack_geometry)
        pack_geometry.update({'cell_to_wall_xyz': [params[0],params[1]], 'pack_xyz': [params[3], params[4]]})
        c, k ,rho = self.extract_pack_materials(params[2])
        thermal_solver = ThermalSolver(self.cells, {'c': c, 'k': k, 'rho': rho}, pack_geometry)
        thermal_solver.create_mesh(pack_geometry['mesh_xy'],method='number_cells', regular_mesh=False)
        thermal_solver._place_cell_in_pack()
        thermal_solver.cell_to_mesh()
        dx = thermal_solver.dx
        dy = thermal_solver.dy
        thermal_solver.T = np.ones(np.shape(thermal_solver.mesh.active)) * self.T_i

        return thermal_solver, c, k, rho, dx, dy

    def prepare_solve(self):
        @jax.jit
        def run_thermal_sim(T_init, Q_dot, pack_params, time_steps):
            step_fn = lambda T_carry, _: self.heat_step_explicit(T_carry, pack_params, Q_dot)
            T_last, T_all = jax.lax.scan(step_fn, T_init, time_steps)
            return jnp.max(T_last)

        return jax.vmap(run_thermal_sim, in_axes=(None, 0,0,None))

    def extract_pack_materials(self,i):
        if i == 0:
            c = 1000
            k = 50
            rho = 2500
        elif i == 1:
            c = 385
            k = 30
            rho = 90
        elif i == 2:
            c = 1053
            k = 0.3
            rho = 1890
        elif i == 3:
            c = 795
            k = 65
            rho = 2318
        else:
            raise ValueError('No pack material selected')
        return c, k, rho


    def best_params(self):
            self.selection()
            params = self.elite[0].read_params()
            return params

    def plot_results(self, best_params):
        ts, c, k, rho, dx, dy = self.init_thermal_solve(best_params)
        dt = 0.2 / (k / (rho * c) * (1 / (dx ** 2) + 1 / (dy ** 2)))
        thickness_z = 0.008
        active_volume = jnp.sum(ts.mesh.generation_cells) * dx * dy * thickness_z
        volumetric_q = 0.35 / active_volume
        Q_dot_i = ts.mesh.generation_cells * volumetric_q
        ts.plot()





