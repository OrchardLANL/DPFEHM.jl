using Test
import DifferentiableBackwardEuler
import DPFEHM
import PyPlot
import Zygote

mins = [0, 0]; maxs = [3, 2]#size of the domain, in meters
ns = [30, 20]#number of nodes on the grid
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)#build the grid
coords = vcat(coords, zeros(size(coords, 2))')#turn it into a 3d coords

c = 1.0
tspan = [0.0, 1.0 / abs(c)]
u0 = map(i->coords[1, i] < 1 / 2  && coords[1, i] > 0 && coords[2, i] > 0.25 && coords[2, i] < 0.75 ? 1.0 : 0.0, 1:size(coords, 2)) + map(i->exp(-((coords[1, i] - 1.0) ^ 2 + (coords[2, i] - 1.0) ^ 2) * 64), 1:size(coords, 2))
vxs = map(neighbor->c, neighbors)
vys = map(neighbor->c * 0.5, neighbors)
vzs = zeros(length(neighbors))
Ds = 1e-2 * ones(length(neighbors))
Qs = zeros(size(coords, 2))
dirichletnodes = [collect(1:ns[2]); collect(size(coords, 2) - ns[2] + 1:size(coords, 2))]
dirichletus = zeros(size(coords, 2))
f(u, p, t) = DPFEHM.transport_residuals(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_u(u, p, t) = DPFEHM.transport_u(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_p(u, p, t) = DPFEHM.transport_vxs(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_t(u, p, t) = zeros(length(u))
function solveforu(vxs)
	DifferentiableBackwardEuler.steps(u0, f, f_u, f_p, f_t, vxs, tspan[1], tspan[2]; abstol=1e-4, reltol=1e-4)
end
@time u_implicit = solveforu(vxs)

fig, axs = PyPlot.subplots(2, 1; figsize=(8, 8))
img = axs[1].imshow(reshape(u0, reverse(ns)...), extent=[mins[1], maxs[1], mins[2], maxs[2]], origin="lower")
fig.colorbar(img, ax=axs[1])
img = axs[2].imshow(reshape(u_implicit[:, end], reverse(ns)...), extent=[mins[1], maxs[1], mins[2], maxs[2]], origin="lower")
fig.colorbar(img, ax=axs[2])
fig.tight_layout()
display(fig)
PyPlot.println()
PyPlot.close(fig)

_, gradient_node = findmin(map(i->sum((coords[:, i] .- [1.5, 1.0, 0.0]) .^ 2), 1:size(coords, 2)))
g = x->solveforu(x)[gradient_node, end]
@time gradg = Zygote.gradient(g, vxs)[1]
function checkgradientquickly(f, x0, gradf, n; delta::Float64=1e-8, kwargs...)
	indicestocheck = sort(collect(1:length(x0)), by=i->abs(gradf[i]), rev=true)[1:n]
	f0 = f(x0)
	for i in indicestocheck
		x = copy(x0)
		x[i] += delta
		fval = f(x)
		grad_f_i = (fval - f0) / delta
		@test isapprox(gradf[i], grad_f_i; kwargs...)
	end
end
checkgradientquickly(g, vxs, gradg, 3; atol=1e-4, rtol=1e-3)

