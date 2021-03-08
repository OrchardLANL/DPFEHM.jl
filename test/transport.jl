using Test
import DifferentiableBackwardEuler
import DPFEHM
import Zygote

doplot = false
if doplot == true
	import PyPlot
end

mins = [0]; maxs = [4]#size of the domain, in meters
ns = [4000]#number of nodes on the grid
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid1d(mins, maxs, ns, 1.0, 1.0)#build the grid
coords = vcat(coords, zeros(size(coords, 2))', zeros(size(coords, 2))')#turn it into a 3d coords

c = 1.0
tspan = [0.0, 1.0]
sigma = 3e-1
u0 = map(i->exp(-((coords[1, i] - 1.0) ^ 2) / (2 * sigma ^ 2)) / (sigma * sqrt(2 * pi)), 1:size(coords, 2))
vxs = map(neighbor->c, neighbors)
vys = zeros(length(neighbors))
vzs = zeros(length(neighbors))
Ds = 1e-9 * ones(length(neighbors))
Qs = zeros(size(coords, 2))
dirichletnodes = [1, ns[1]]
dirichletus = zeros(ns[1])
f(u, p, t) = DPFEHM.transport_residuals(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_u(u, p, t) = DPFEHM.transport_u(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_p(u, p, t) = DPFEHM.transport_vxs(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_t(u, p, t) = zeros(length(u))
function solveforu(vxs)
	DifferentiableBackwardEuler.steps(u0, f, f_u, f_p, f_t, vxs, tspan[1], tspan[2]; abstol=1e-4, reltol=1e-4)
end
print("transport forward")
@time u = solveforu(vxs)
u1 = u[:, end]

@test isapprox(sum(u1 .* volumes), sum(u0 .* volumes); rtol=1e-2)#test conservation of mass
@test isapprox(1.0, sum(u0 .* coords[1, :] .* volumes); rtol=1e-2)#check that the initial condition is putting mass in the right place
@test isapprox(2.0, sum(u1 .* coords[1, :] .* volumes); rtol=1e-2)#check that the advection is moving the mass
@test isapprox(sigma ^ 2, sum(u0 .* (coords[1, :] .- 1) .^ 2 .* volumes); rtol=1e-2)#check that the variance is what we set it to 
@test isapprox(sigma ^ 2, sum(u1 .* (coords[1, :] .- 2) .^ 2 .* volumes); rtol=1e-1)#check that the variance is what we set it to  -- note there will be some numerical dispersion

if doplot
	fig, ax = PyPlot.subplots()
	ax.plot(coords[1, :], u0, label="u(0)")
	ax.plot(coords[1, :], u1, label="u(1)")
	ax.legend()
	fig.tight_layout()
	display(fig)
	PyPlot.println()
	PyPlot.close(fig)
end

_, gradient_node = findmin(map(i->sum((coords[:, i] .- [1.5, 1.0, 0.0]) .^ 2), 1:size(coords, 2)))
g = x->solveforu(x)[gradient_node, end]
print("transport gradient")
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
