using LaTeXStrings
import AlgebraicMultigrid
import DelimitedFiles
import DPFEHM
import FEHM
import HDF5
import ILUZero
import IterativeSolvers
import JLD
import Preconditioners
import PyPlot

include("utilities.jl")

network_dirs = ["thomas_tpl_p2_x01", "thomas_tpl_p3_x01", "thomas_tpl_p5_x01", "thomas_tpl_p10_x01", "thomas_tpl_p20_x01"]
network_dirs = [network_dirs; ["var_b_0.0", "var_b_0.5", "var_b_1.0"]]
network_dirs = [network_dirs; ["TSA250_50"]]
need_to_download_data = false
for dir in network_dirs
	if !isdir(dir)
		need_to_download_data = true
	end
end
if need_to_download_data
	download("https://zenodo.org/record/5213727/files/results.tar.gz?download=1", "./results.tar.gz")
	run(`tar xzf results.tar.gz`)
	download("https://zenodo.org/record/5213727/files/meshes.tar.gz?download=1", "./meshes.tar.gz")
	run(`tar xzf meshes.tar.gz`)
end
times = Dict()
chs = Dict()
for (i, network_dir) in enumerate(network_dirs)
	resultsfilename = "results/" * replace(network_dir, "/"=>"_") * "_results.jld"
	if isfile(resultsfilename)
		thesetimes, thesechs = JLD.load(resultsfilename, "times", "chs")
		times[network_dir] = thesetimes
		chs[network_dir] = thesechs
	else
		times[network_dir] = Dict()
		chs[network_dir] = Dict()
	end
	@time A, b, others = loaddir(network_dir)
	for (j, (solver_name, solver)) in enumerate(preconditioned_solvers)
		for (k, (preconditioner_name, preconditioner)) in enumerate(preconditioners)
			combo_name = string(solver_name, "+", preconditioner_name)
			if !haskey(times[network_dir], combo_name)
				t = @elapsed begin
					pl = preconditioner(A)
					x, ch = solver(A, b, pl)
				end
				ch = ch / ch[end] * sqrt(sum((A * x .- b) .^ 2) / sum(b .^ 2))
				@show combo_name, t
				times[network_dir][combo_name] = t
				chs[network_dir][combo_name] = ch
				JLD.save(resultsfilename, "times", times[network_dir], "chs", chs[network_dir])
			end
		end
	end
	for (j, (solver_name, solver)) in enumerate(unpreconditioned_solvers)
		if !haskey(times[network_dir], solver_name)
			t = @elapsed begin
				x, ch = solver(A, b)
			end
			@show solver_name, t
			ch = ch / ch[end] * sqrt(sum((A * x .- b) .^ 2) / sum(b .^ 2))
			times[network_dir][solver_name] = t
			chs[network_dir][solver_name] = ch
			JLD.save(resultsfilename, "times", times[network_dir], "chs", chs[network_dir])
		end
	end
end
