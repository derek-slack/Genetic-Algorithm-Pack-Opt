import jax
import jax.numpy as jnp
import numpy as np
import copy


class Chromosome:
    def __init__(self,parameters, pack_geometry):
        self.process_parameters(parameters)
        self.pack_geometry = pack_geometry
    def process_parameters(self, parameters: dict):
        for key in parameters.keys():
            self.test = parameters[key]
            self.parameters = parameters

    def encode(self):

        parameters_encoded = copy.deepcopy(self.parameters)
        for param in self.parameters.keys():

            M =self.parameters[param].min_bit
            if type(self.parameters[param].max_bit) is str:
                if self.parameters[param].max_bit == 'L':
                    n_ind = 0
                elif self.parameters[param].max_bit == 'W':
                    n_ind = 1
                X = self.parameters[self.parameters[param].max_bit].value/2 - (self.pack_geometry['cell_n_xyz'][n_ind]-1)*(self.pack_geometry['cell_rz'][0]+self.pack_geometry['Buffer'])
                V = (X-M)* (self.parameters[param].value) + M
            else:
                V = self.parameters[param].value
                X = self.parameters[param].max_bit

            n = self.parameters[param].n_bit
            param_encode = np.binary_repr(int(np.ceil((V-M)/(X-M)*(2**n-1))),width = self.parameters[param].n_bit)
            parameters_encoded[param].value = param_encode
        self.parameters_encoded = parameters_encoded

    def read_params(self):
        param_array = np.zeros(len(self.parameters_encoded))
        for i, param in enumerate(self.parameters.keys()):
            V = self.parameters_encoded[param].value
            M = self.parameters_encoded[param].min_bit
            if type(self.parameters[param].max_bit) is str:
                if self.parameters[param].max_bit == 'L':
                    n_ind = 0
                elif self.parameters[param].max_bit == 'W':
                    n_ind = 1
                X = self.parameters[self.parameters[param].max_bit].value / 2 - (
                            self.pack_geometry['cell_n_xyz'][n_ind] - 1) * (
                                self.pack_geometry['cell_rz'][0] + self.pack_geometry['Buffer'])

            else:
                X = self.parameters[param].max_bit
            n = self.parameters_encoded[param].n_bit

            param_real = (int(V,2)/(2**n - 1))*(X-M) + M
            self.parameters[param].value = param_real
            param_array[i] = param_real
        return param_array

    def evaluate(self):
        pass

