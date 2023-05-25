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

from typing import Tuple

import numpy
import torch


class Mesh2d:
    def __init__(self, points: numpy.ndarray, faces: numpy.ndarray, boundary: numpy.ndarray,
                 requires_grad: bool = True):
        assert points.ndim == 3 and points.shape[1] == 2 and points.shape[2] >= 1
        self.num_points = points.shape[0]
        self.num_params = points.shape[2] - 1
        self.parameters = torch.zeros((self.num_params, ),
                                      dtype=torch.float32,
                                      requires_grad=requires_grad)

        points = torch.from_numpy(points).float()
        self.points = points[:, :, 0] + \
            torch.einsum("ijk,k->ij", points[:, :, 1:], self.parameters)

        assert faces.dtype == int and faces.ndim == 2 and faces.shape[1] == 3
        assert numpy.all(numpy.logical_and(
            0 <= faces, faces < self.num_points))
        self.faces = torch.from_numpy(faces)
        self.num_faces = self.faces.shape[0]

        assert boundary.dtype == int and boundary.ndim == 1 and boundary.shape[
            0] == self.num_points
        self.boundary = boundary
        self.num_boundaries = max(boundary)

    @staticmethod
    def load(filename):
        """Loads a numpy npz file saved by diffmesh."""
        with numpy.load(filename) as data:
            return Mesh2d(data["points"], data["faces"], data["boundary"])

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns the bounding box of all points."""
        mins, _ = torch.min(self.points.detach(), dim=0)
        maxs, _ = torch.max(self.points.detach(), dim=0)
        return (mins[0].item(), mins[1].item(), maxs[0].item(), maxs[1].item())

    def print_info(self):
        """Prints statistics about this 2d mesh object."""
        print("num_params:", self.num_params)
        print("num_points:", self.num_points)
        print("num_faces:", self.num_faces)
        print("num_boundaries:", self.num_boundaries)
        print("points.shape:", tuple(self.points.shape))
        print("faces.shape:", tuple(self.faces.shape))
        print("bounding_box:", self.bounding_box())

    def face_coords(self) -> torch.Tensor:
        """
        Returns a tensor of shape [num_faces, 3, 2] containing the
        coordinates of the points of all faces.
        """
        coords = torch.index_select(self.points, 0, self.faces.flatten())
        return torch.reshape(coords, (self.num_faces, 3, 2))

    @staticmethod
    def face_matrices(face_coords: torch.Tensor) -> torch.Tensor:
        """
        Returns the transformation matrix for each face that convert the
        elementary triangles to their real coordinates.
        """
        return face_coords[:, 1:, :] - face_coords[:, 0, :].unsqueeze(1)
