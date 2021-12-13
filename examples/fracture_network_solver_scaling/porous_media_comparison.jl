using LaTeXStrings
import AlgebraicMultigrid
import DPFEHM
import IterativeSolvers
import JLD
import LinearAlgebra
import PyPlot

function solve_cg_amg(A, b)
	Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(A))
	x, ch = IterativeSolvers.cg(A, b; maxiter=10000, reltol=1e-12, log=true, Pl=Pl)
	return x, ch.data[:resnorm]
end

function solve_cholesky(A, b)
	Af = LinearAlgebra.cholesky(A)
	x = Af \ b
	return x, [sqrt(sum((A * x .- b) .^ 2) / sum(b .^ 2))]
end

function setup_problem(n)
	mins = [0, 0, 0]#meters
	maxs = [100, 100, 100]#meters
	ns = [n, n, n]
	num_eigenvectors = 200
	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid3d(mins, maxs, ns)
	Qs = zeros(size(coords, 2))
	dirichletnodes = Int[]
	dirichleths = zeros(size(coords, 2))
	for i = 1:size(coords, 2)
		if coords[1, i] == mins[1]
			push!(dirichletnodes, i)
			dirichleths[i] = 1.0
		elseif coords[1, i] == maxs[1]
			push!(dirichletnodes, i)
			dirichleths[i] = 0.0
		end
	end
	Ks = ones(length(neighbors))
	args = (zeros(length(Qs) - length(dirichletnodes)), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args...)
	A = DPFEHM.groundwater_h(args...)
	return A, b, (neighbors, areasoverlengths, Ks, Qs, dirichletnodes, dirichleths)
end

for i = 1:10#this loop gets compilation out of the way
	A, b, _ = setup_problem(2 ^ 2)
	solve_cholesky(A, b)
	solve_cg_amg(A, b)
end
if !isfile("porous_media_results.jld")
	ns = 2 .^ (2:7)
	num_solves = 5
	cholesky_times = zeros(length(ns), num_solves)
	cg_times = zeros(length(ns), num_solves)
	for (j, n) in enumerate(ns)
		A, b = setup_problem(n)
		these_cholesky_times = Float64[]
		for i = 1:num_solves
			t = @elapsed solve_cholesky(A, b)
			cholesky_times[j, i] = t
			t = @elapsed solve_cg_amg(A, b)
			cg_times[j, i] = t
		end
	end
	JLD.save("porous_media_results.jld", "cholesky_times", cholesky_times, "cg_times", cg_times, "ns", ns)
end

cholesky_times, cg_times, ns = JLD.load("porous_media_results.jld", "cholesky_times", "cg_times", "ns")

fig, ax = PyPlot.subplots()
ax.loglog(ns .^ 3, Statistics.median(cholesky_times; dims=2), ".", label="Cholesky")
ax.loglog(ns .^ 3, Statistics.median(cg_times; dims=2), ".", label="CG+AMG")
ax.legend()
display(fig)
println()
PyPlot.close(fig)
