using LaTeXStrings
import AlgebraicMultigrid
import DPFEHM
import IterativeSolvers
import JLD
import LinearAlgebra
import PyPlot
import Statistics

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
	num_solves = 10
	cholesky_times = zeros(length(ns), num_solves)
	cg_times = zeros(length(ns), num_solves)
	for (j, n) in enumerate(ns)
		A, b = setup_problem(n)
		these_cholesky_times = Float64[]
		for i = 1:num_solves
			@show n, i
			t = @elapsed solve_cholesky(A, b)
			@show t
			cholesky_times[j, i] = t
			t = @elapsed solve_cg_amg(A, b)
			@show t
			cg_times[j, i] = t
		end
	end
	JLD.save("porous_media_results.jld", "cholesky_times", cholesky_times, "cg_times", cg_times, "ns", ns)
end

function plotit(cholesky_times, cg_times, ns, ax)
	ax.loglog(ns, Statistics.median(cholesky_times; dims=2), ".", label="Cholesky", color="C0", ms=10, alpha=0.75)
	ax.loglog(ns, Statistics.median(cg_times; dims=2), ".", label="CG+AMG", color="C1", ms=10, alpha=0.75)
	for i = 1:size(cholesky_times, 1)
		ax.plot([ns[i], ns[i]], sort(cholesky_times[i, :])[[2, end - 1]], color="C0", alpha=0.75)
		ax.plot([ns[i], ns[i]], sort(cg_times[i, :])[[2, end - 1]], color="C1", alpha=0.75)
	end
	ax.set(xlabel="Degrees of freedom", ylabel="Time [s]")
	ax.legend()
end

cholesky_times, cg_times, ns = JLD.load("porous_media_results.jld", "cholesky_times", "cg_times", "ns")
fig, axs = PyPlot.subplots(1, 3; figsize=(12, 4))
plotit(cholesky_times, cg_times, ns .^ 3, axs[3])

function setup_problem_2d(n)
	mins = [0, 0]#meters
	maxs = [100, 100]#meters
	ns = [n, n]
	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
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
	A, b, _ = setup_problem_2d(2 ^ 2)
	solve_cholesky(A, b)
	solve_cg_amg(A, b)
end
if !isfile("porous_media_results_2d.jld")
	ns = 2 .^ (3:13)
	num_solves = 10
	cholesky_times = zeros(length(ns), num_solves)
	cg_times = zeros(length(ns), num_solves)
	for (j, n) in enumerate(ns)
		A, b = setup_problem_2d(n)
		these_cholesky_times = Float64[]
		for i = 1:num_solves
			@show n, i
			t = @elapsed solve_cholesky(A, b)
			@show t
			cholesky_times[j, i] = t
			t = @elapsed solve_cg_amg(A, b)
			@show t
			cg_times[j, i] = t
		end
	end
	JLD.save("porous_media_results_2d.jld", "cholesky_times", cholesky_times, "cg_times", cg_times, "ns", ns)
end

cholesky_times, cg_times, ns = JLD.load("porous_media_results_2d.jld", "cholesky_times", "cg_times", "ns")
plotit(cholesky_times, cg_times, ns .^ 2, axs[2])

function setup_problem_1d(n)
	mins = [0]#meters
	maxs = [100]#meters
	ns = [n]
	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid1d(mins, maxs, ns, 1.0, 1.0)
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
	A, b, _ = setup_problem_1d(2 ^ 2)
	solve_cholesky(A, b)
	solve_cg_amg(A, b)
end
if !isfile("porous_media_results_1d.jld")
	ns = 10 .^ (1:8)
	num_solves = 10
	cholesky_times = zeros(length(ns), num_solves)
	cg_times = zeros(length(ns), num_solves)
	for (j, n) in enumerate(ns)
		A, b = setup_problem_1d(n)
		these_cholesky_times = Float64[]
		for i = 1:num_solves
			@show n, i
			t = @elapsed solve_cholesky(A, b)
			@show t
			cholesky_times[j, i] = t
			t = @elapsed solve_cg_amg(A, b)
			@show t
			cg_times[j, i] = t
		end
	end
	JLD.save("porous_media_results_1d.jld", "cholesky_times", cholesky_times, "cg_times", cg_times, "ns", ns)
end

cholesky_times, cg_times, ns = JLD.load("porous_media_results_1d.jld", "cholesky_times", "cg_times", "ns")
plotit(cholesky_times, cg_times, ns, axs[1])
for i = 1:3
	axs[i].set_title("$(i)D")
end
fig.tight_layout()
display(fig)
println()
fig.savefig("porous_media_comparison.pdf")
PyPlot.close(fig)
