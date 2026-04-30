import timeit

import jax.lax
import numpy as np
from itertools import product
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
class ThermalSolver():
    def __init__(self, cells, heat_params, pack_geometry):

        # Number of cells in x, y, z direction geometrically i.e:
        # [4, 2, 1] - 4 rows of 2 cells in 1 cell width
        self.cells = cells
        self.heat_params = heat_params
        cell_rz, pack_xyz, boundary_conditions, cell_to_wall_xyz, cell_n_xyz = self.unpack_pack_geometry(pack_geometry)
        self.set_pack_geometry(pack_xyz, boundary_conditions, cell_to_wall_xyz, cell_n_xyz)
        self.cell_rz = cell_rz
        self.cell_n_xyz = cell_n_xyz



    def edit_heat_params(self, heat_param, update_val):
        '''
        Updates thermal parameter set values
        :param heat_param: parameter/s to update
        :param update_val: values to update parameter to
        :return: updated parameter set
        '''
        update_params = self.heat_params
        for param_str, param_val in zip(heat_param, update_val):
            update_params[param_str] = param_val
        self.heat_params = update_params
        return update_params

    def unpack_pack_geometry(self, pack_geometry_dict):
        key_options = ['cell_rz', 'pack_xyz', 'cell_to_wall_xyz', 'boundary_conditions', 'cell_n_xyz', 'pack_xyz','2d','mesh_xy','Buffer']
        for k in pack_geometry_dict.keys():
            if k not in key_options:
                raise ValueError(f'variable {k} is not an option for pack geometry. Must be one of {key_options}')
        if 'cell_to_wall_xyz' not in pack_geometry_dict.keys():
            raise ValueError('Must define `cell_to_wall_xyz` to create pack, this is the distances from the wall to the cells\n'
                             'closest to the wall. All other cells in the same line are evenly spaced. formatted in:\n'
                             ' [`x distance to wall`, `y distance to wall`, `z distance to wall`] (m)')
        if 'cell_n_xyz' not in pack_geometry_dict.keys():
            raise ValueError(f'Must define `cell_n_xyz` this is the number of cells in each dimension geometrically')
        default_keys = {'cell_rz': [0.18, 0.065], 'pack_xyz':[0.2,0.2,0.2], 'boundary_conditions':[], '2d':False}
        default_keys.update(pack_geometry_dict)

        cell_rz = default_keys['cell_rz']
        pack_xyz = default_keys['pack_xyz']
        boundary_conditions = default_keys['boundary_conditions']
        cell_to_wall_xyz = default_keys['cell_to_wall_xyz']
        cell_n_xyz = default_keys['cell_n_xyz']
        self.TwoD = default_keys['2d']
        if self.TwoD:
            self.dim = 2
        if not self.TwoD:
            self.dim = 3

        return cell_rz, pack_xyz, boundary_conditions, cell_to_wall_xyz, cell_n_xyz

    def set_pack_geometry(self, pack_xyz, boundary_conditions, cell_to_wall_xyz, cell_n_xyz):
        self.pack_xyz = pack_xyz
        if cell_to_wall_xyz == 'even':
            c_empty = np.zeros(len(pack_xyz))
            for dim in range(len(pack_xyz)):
                n_cells = cell_n_xyz[dim]
                d_dim = pack_xyz[dim]/(n_cells+1)
                c_empty[dim] = d_dim
            self.cell_to_wall_xyz = c_empty
        else:
            self.cell_to_wall_xyz = cell_to_wall_xyz
        self.boundary_conditions = boundary_conditions

        if self.TwoD:
            self.pack_width = pack_xyz[0]
            self.pack_length = pack_xyz[1]
            # print(f"2D Pack created: \n Width (m): {self.pack_width} \n Length (m): {self.pack_length} \n")
        else:
            self.pack_width = pack_xyz[0]
            self.pack_length = pack_xyz[1]
            self.pack_height = pack_xyz[2]
            print(f"Pack created: \n Width (m): {self.pack_width} \n Length (m): {self.pack_length} \n Height (m): {self.pack_height}")



    def create_mesh(self, mesh_xyz, method = "dx", regular_mesh = True):
        if self.TwoD:
            self.create_mesh_2D(mesh_xyz, method=method, regular_mesh=regular_mesh)
            return
        mesh_created = False

        if method == "dx":
            if regular_mesh:
                dx = mesh_xyz
                dy = dx
                dz = dx
            else:
                dx = mesh_xyz[0]
                dy = mesh_xyz[1]
                dz = mesh_xyz[2]

        elif method == "number_cells":
            if regular_mesh:
                dx = self.pack_width/mesh_xyz
                dy = dx
                dz = dx
            else:
                dx = self.pack_width/mesh_xyz[0]
                dy = self.pack_length/mesh_xyz[1]
                dz = self.pack_height/mesh_xyz[2]

        elif method == "explicit":
            dx = mesh_xyz[0][1] - mesh_xyz[0][0]
            dy = mesh_xyz[1][1] - mesh_xyz[1][0]
            dz = mesh_xyz[2][1] - mesh_xyz[2][0]
            mesh_created = True

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.cell_volume = dx*dy*dz

        if mesh_created:
            mesh = mesh_xyz
        else:
            nx = np.ceil(self.pack_width/dx).astype(int)
            ny = np.ceil(self.pack_length/dy).astype(int)
            nz = np.ceil(self.pack_height/dz).astype(int)

            mesh = np.stack(np.meshgrid(
                np.arange(nx) * dx,
                np.arange(ny) * dy,
                np.arange(nz) * dz,
                indexing='ij'
            ), axis=-1)

        self.mesh = ThermalMesh(mesh)


        self.mesh_x = np.linspace(0, self.pack_width, nx)
        self.mesh_y = np.linspace(0, self.pack_length, ny)
        self.mesh_z = np.linspace(0, self.pack_height, nz)

        return mesh

    def create_mesh_2D(self, mesh_xy, method="dx", regular_mesh=True):
        mesh_created = False

        if method == "dx":
            if regular_mesh:
                dx = mesh_xy
                dy = dx

            else:
                dx = mesh_xy[0]
                dy = mesh_xy[1]
        elif method == "number_cells":
            if regular_mesh:
                dx = self.pack_width/mesh_xy
                dy = dx

            else:
                dx = self.pack_width/mesh_xy[0]
                dy = self.pack_length/mesh_xy[1]

        elif method == "number_cells":
            if regular_mesh:
                dx = self.pack_width / mesh_xy
                dy = dx
            else:
                dx = self.pack_width / mesh_xy[0]
                dy = self.pack_length / mesh_xy[1]

        elif method == "explicit":
            dx = mesh_xy[0][1] - mesh_xy[0][0]
            dy = mesh_xy[1][1] - mesh_xy[1][0]

            mesh_created = True

        self.dx = dx
        self.dy = dy

        self.cell_volume = dx * dy

        if mesh_created:
            mesh = mesh_xy
        else:
            if method == 'number_cells':
                if regular_mesh:
                    nx = mesh_xy
                    ny = nx
                else:
                    nx = mesh_xy[0]
                    ny = mesh_xy[1]

            else:
                nx = np.ceil(self.pack_width / dx).astype(int)
                ny = np.ceil(self.pack_length / dy).astype(int)

            mesh = np.stack(np.meshgrid(
                np.arange(nx) * dx,
                np.arange(ny) * dy,
                indexing='ij'
            ), axis=-1)

        self.mesh = ThermalMesh(mesh, TwoD=True)

        self.mesh_x = np.linspace(0, self.pack_width, nx)
        self.mesh_y = np.linspace(0, self.pack_length, ny)

        return mesh

    def _place_cell_in_pack(self):
        # space_between_cells = (self.pack_xyz - 2*self.cell_to_wall_xyz)/(self.cell_n_xyz - np.ones(3))
        cell_pos = []

        for dir in range(self.dim):
            cell_pos.append(np.linspace(self.cell_to_wall_xyz[dir], (self.pack_xyz[dir]-self.cell_to_wall_xyz[dir]), self.cell_n_xyz[dir]))
        if self.TwoD:
            cell_pos_combs = list(product(cell_pos[0], cell_pos[1]))
        else:
            cell_pos_combs = list(product(cell_pos[0], cell_pos[1], cell_pos[2]))

        count_cell = 0

        for branch in self.cells:
            for cell in branch:
                cell.position = np.array((list(cell_pos_combs[count_cell]))).reshape(-1,1)
                cell.count = count_cell
                count_cell += 1

    def cell_to_mesh(self):
        if self.TwoD:
            self.cell_to_mesh_2D()
            return
        for branch in self.cells:
            for cell in branch:

                # to centers
                dx_c = self.mesh.mesh_points[:, :, :, 0] - cell.position[0]
                dy_c = self.mesh.mesh_points[:, :, :, 1] - cell.position[1]
                dz_c = self.mesh.mesh_points[:, :, :, 2] - cell.position[2]
                dr_c = np.sqrt(dx_c**2 + dy_c**2)

                # dist to bound
                dr_b = dr_c - self.cell_rz[0]  # distance to border
                dz_b = (abs(dz_c) - (self.cell_rz[1])/2)  # distance to border


                d_gen = 0.001 # Generation term

                # Outer side surface
                side_wall = (abs(dr_b) <= d_gen) & (abs(dz_c) <= self.cell_rz[1] / 2)

                # Top and bottom caps
                end_caps = (abs(dz_b) <= d_gen) & (dr_c <= self.cell_rz[0])

                # Combine
                all_gen = side_wall | end_caps
                all_bound = ((dr_c < self.cell_rz[0]) & (abs(dz_c) < self.cell_rz[1] / 2)) & (~all_gen)
                all_active = (~all_gen) & (~all_bound)

                cell.generation_cells = all_gen
                cell.bound_cells = all_bound
                cell.all_active = all_active

                self.mesh.generation_cells |= all_gen
                self.mesh.boundary_cell |= all_bound
                self.mesh.active |= all_active

    def cell_to_mesh_2D(self):

        for branch in self.cells:
            for cell in branch:

                # to centers
                dx_c = self.mesh.mesh_points[:, :, 0] - cell.position[0]
                dy_c = self.mesh.mesh_points[:, :, 1] - cell.position[1]
                dr_c = np.sqrt(dx_c**2 + dy_c**2)

                # dist to bound
                dr_b = dr_c - self.cell_rz[0]  # distance to border

                d_gen = self.dx/0.9 # Generation term

                # Outer side surface
                side_wall = (dr_b >= -d_gen) & (dr_b <= 0)

                # Combine
                all_gen = side_wall
                all_bound = ((dr_c < self.cell_rz[0])) & (~all_gen)
                all_active = (~all_gen) & (~all_bound)

                cell.generation_cells = all_gen
                cell.bound_cells = all_bound
                cell.all_active = all_active

                self.mesh.generation_cells |= all_gen
                self.mesh.boundary_cell |= all_bound

        all_active = (~self.mesh.generation_cells) & (~self.mesh.boundary_cell)

        self.mesh.active |= all_active

    def calculate_Q_dot(self, Q_dot_model):
        count_cell = 0
        Q_dot_cell = []
        for i, branch in enumerate(self.cells):
            for j, cell in enumerate(branch):
                # Q_dot_cell.append(Q_dot_model[i][j]/(sum(sum(sum(cell.generation_cells)))*self.cell_volume))
                Q_dot_cell.append(Q_dot_model[i][j] / (self.cell_volume))
                cell.q_dot_t = Q_dot_cell[count_cell]
                count_cell += 1
        return Q_dot_cell

    def Q_dot_to_matrix(self, Q_dot_cell, cell, empty_set):
        return (Q_dot_cell)*(cell.generation_cells + empty_set)

    def compile_Q_dot(self, Q_dot_cell):
        if self.TwoD:
            Q_dot_mesh = np.zeros(self.mesh.mesh_points[:, :, 0].shape)
        else:
            Q_dot_mesh = np.zeros(self.mesh.mesh_points[:, :, :, 0].shape)
        k=0
        for branch in self.cells:
            for cell in branch:
                Q_dot_mesh += self.Q_dot_to_matrix(Q_dot_cell[k],cell, np.zeros(Q_dot_mesh))
                k+=1
        return Q_dot_mesh

    def laplacian_bc_matrix(self):
        pass

    def _print_volume(self):
        pack_volume = self.pack_xyz[0]*self.pack_xyz[1]*self.pack_xyz[2]
        print(f'Pack volume is {pack_volume}')




    def create_heat_vmap(self):
        pass

    def update_average_temperature(self, T_next):
        for branch in self.cells:
            for cell in branch:
                update_temp = cell.generation_cells * T_next
                average_temperature = np.mean(update_temp[cell.generation_cells])
                cell.average_temperature = average_temperature

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        im = ax.imshow(np.array(self.mesh.boundary_cell)*100 + np.array(self.mesh.generation_cells) * 200, cmap='hot', interpolation='nearest')
        plt.show()
class ThermalMesh:
    def __init__(self, mesh_points, TwoD=False):

        self.mesh_points = mesh_points
        if TwoD:
            shape = self.mesh_points.shape[:2]
        else:
            shape = self.mesh_points.shape[:3]  # (Nx, Ny, Nz)

        self.generation_cells = np.zeros(shape, dtype=bool)
        self.boundary_cell = np.zeros(shape, dtype=bool)
        self.active = np.zeros(shape, dtype=bool)

