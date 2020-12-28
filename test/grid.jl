using Test
import DPFEHM

ns = [5, 10, 20]
mins = [0, 0, 0]
maxs = [1, 1, 1]

#test that the 2d grid first goes over the the y coordinate, then the x coordinate
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins[1:2], maxs[1:2], ns[1:2], maxs[3] - mins[3])
@test coords[2, 1] != coords[2, 2]
@test coords[2, 1] == coords[2, 1 + ns[2]]
@test coords[1, 1] != coords[1, 1 + ns[2]]
@test all(coords[1, 1] .== coords[1, 1:ns[2]])
@test sum(volumes) ≈ prod(ns[1:2]) * prod((maxs[1:2] .- mins[1:2]) ./ (ns[1:2] .- 1)) * (maxs[3] - mins[3])#it goes a little higher than just prod(maxs - mins), because the center of the boundary cells are on the boundary, and half of those cells are outside the boundary

#test that the 3d grid first goes over the z coordinate, then the y coordinate, then the x coordinate
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid3d(mins, maxs, ns)
@test coords[3, 1] != coords[3, 2]
@test coords[3, 1] == coords[3, 1 + ns[3]]
@test coords[2, 1] != coords[2, 1 + ns[3]]
@test all(coords[2, 1] .== coords[2, 1:ns[3]])
@test coords[1, 1] != coords[1, 1 + ns[3] * ns[2]]
@test all(coords[1, 1] .== coords[1, 1:ns[2] * ns[3]])
@test sum(volumes) ≈ prod(ns) * prod((maxs .- mins) ./ (ns .- 1))#it goes a little higher than just prod(maxs - mins), because the center of the boundary cells are on the boundary, and half of those cells are outside the boundary

#test that it is computing the darcy velocity correctly for a constant pressure gradient along each of the coordinate axes
Ks = ones(reverse(ns)...)
h = [x for z = 1:ns[3], y = 1:ns[2], x = 1:ns[1]]
dv = DPFEHM.darcy_velocity(h, Ks, mins, maxs, ns)
@test dv(0.5, 0.5, 0.5) ≈ [ns[1] - 1, 0, 0]
h = [y for z = 1:ns[3], y = 1:ns[2], x = 1:ns[1]]
dv = DPFEHM.darcy_velocity(h, Ks, mins, maxs, ns)
@test dv(0.5, 0.5, 0.5) ≈ [0, ns[2] - 1, 0]
h = [z for z = 1:ns[3], y = 1:ns[2], x = 1:ns[1]]
dv = DPFEHM.darcy_velocity(h, Ks, mins, maxs, ns)
@test dv(0.5, 0.5, 0.5) ≈ [0, 0, ns[3] - 1]
