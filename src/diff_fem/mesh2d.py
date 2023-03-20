# Copyright (C) 2023, Miklos Maroti
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
import torch


class Mesh2d:
    def __init__(self, points: numpy.ndarray, faces: numpy.ndarray, boundary: numpy.ndarray):
        assert points.ndim == 3 and points.shape[1] == 2 and points.shape[2] >= 1
        self.num_points = points.shape[0]
        self.num_params = points.shape[2] - 1
        self.parameters = torch.zeros((self.num_params, ),
                                      dtype=torch.float32,
                                      requires_grad=True)

        points = torch.from_numpy(points).float()
        self.points = points[:, :, 0] + \
            torch.einsum("ijk,k->ij", points[:, :, 1:], self.parameters)

        assert faces.dtype == int and faces.ndim == 2 and faces.shape[1] == 3
        for idx in faces.flatten():
            assert 0 <= idx < self.num_points
        self.faces = faces

        assert boundary.dtype == int and boundary.ndim == 1 and boundary.shape[
            0] == self.num_points
        self.boundary = boundary
        self.num_boundaries = max(boundary)

    @staticmethod
    def load(filename):
        with numpy.load(filename) as data:
            return Mesh2d(data["points"], data["faces"], data["boundary"])
