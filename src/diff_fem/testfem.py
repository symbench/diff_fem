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
import matplotlib.pyplot as plt

X1, Y1 = 1.0, 1.0
X2, Y2 = 2.0, 3.0
X3, Y3 = 4.0, 4.0

A = numpy.array([[X2-X1, X3-X1], [Y2-Y1, Y3-Y1]])
B = numpy.linalg.inv(A)
C = numpy.array([X1, Y1])


def plotit(fun):
    xs = numpy.arange(0.0, 5.0, 0.05)
    ys = numpy.arange(0.0, 5.0, 0.05)
    xs, ys = numpy.meshgrid(xs, ys)
    zs = numpy.empty(xs.shape)

    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            z = fun(xs[i, j], ys[i, j])
            zs[i, j] = 0.0 if z is None else z

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.contour3D(xs, ys, zs, 50, cmap='hot')
    surf = ax.plot_surface(xs, ys, zs, cmap='hot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(surf)
    plt.show()


def integrate(fun):
    delta = 0.01
    xs = numpy.arange(0.0, 5.0, delta)
    ys = numpy.arange(0.0, 5.0, delta)
    xs, ys = numpy.meshgrid(xs, ys)
    zs = numpy.empty(xs.shape)

    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            z = fun(xs[i, j], ys[i, j])
            zs[i, j] = 0.0 if z is None else z

    us = (zs[1:, 1:] + zs[:-1, 1:] + zs[1:, :-1] + zs[:-1, :-1]) * 0.25
    return numpy.sum(us) * delta * delta


def elem2norm(x, y):
    u = numpy.array([x, y])
    v = numpy.matmul(A, u) + C
    return v[0], v[1]


def norm2elem(x, y):
    v = numpy.array([x, y])
    u = numpy.matmul(B, v - C)
    return u[0], u[1]


def triangle(x, y):
    x, y = norm2elem(x, y)
    return 1.0 if x >= 0 and y >= 0 and x + y <= 1 else None


def base1(x, y):
    x, y = norm2elem(x, y)
    if x >= 0 and y >= 0 and x + y <= 1:
        return 1.0 - x - y
    return None


def base2(x, y):
    x, y = norm2elem(x, y)
    if x >= 0 and y >= 0 and x + y <= 1:
        return x
    return None


def base3(x, y):
    x, y = norm2elem(x, y)
    if x >= 0 and y >= 0 and x + y <= 1:
        return y
    return None


def deriv1(fun):
    def fun2(x, y):
        z1 = fun(x - 0.01, y)
        z2 = fun(x + 0.01, y)
        if z1 is None or z2 is None:
            return None
        return (z2 - z1) / 0.02
    return fun2


def deriv2(fun):
    def fun2(x, y):
        z1 = fun(x, y - 0.01)
        z2 = fun(x, y + 0.01)
        if z1 is None or z2 is None:
            return None
        return (z2 - z1) / 0.02
    return fun2


def test():
    #print(elem2norm(1, 1))
    #print(norm2elem(2, 2))
    # plotit(deriv2(base1))
    print(integrate(triangle))
    print(0.5 * numpy.linalg.det(A))
