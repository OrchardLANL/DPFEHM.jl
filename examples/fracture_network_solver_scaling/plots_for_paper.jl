using LaTeXStrings
import JLD
import PyPlot

include("utilities.jl")

times = Dict()
chs = Dict()

#do the ugta plot
network_dir = "TSA250_50"
fig, ax = PyPlot.subplots()
resultsfilename = "results/" * replace(network_dir, "/"=>"_") * "_results.jld"
times, chs = JLD.load(resultsfilename, "times", "chs")
names = [map(x->x[1][1] * "+" * x[2][1], Base.product(preconditioned_solvers, preconditioners))[:]; map(x->x[1], unpreconditioned_solvers)]
names = sort(names; by=n->times[n])
ax.barh(names, map(x->times[x], names), color=map(n->ifelse(chs[n][end] < 1e-10, "C0", "C1"), names))
c0_patch = PyPlot.matplotlib.patches.Patch(color="C0", label="converged")
c1_patch = PyPlot.matplotlib.patches.Patch(color="C1", label="did not converge")
ax.legend(handles=[c0_patch, c1_patch], fontsize=14)
ax.set_xscale("log")
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_xlim(1e2, 3e4)
ax.set_xlabel("Run Time (seconds)", fontsize=16)
fig.tight_layout()
fig.savefig("ugta.pdf")
display(fig)
println()
PyPlot.close(fig)

#do the plot on the grid scaling for the thomas dfns
numnodes = Dict("thomas_tpl_p2_x01"=>62070, "thomas_tpl_p3_x01"=>497635, "thomas_tpl_p5_x01"=>2309464, "thomas_tpl_p10_x01"=>7123318, "thomas_tpl_p20_x01"=>13038898)
network_dirs = ["thomas_tpl_p2_x01", "thomas_tpl_p3_x01", "thomas_tpl_p5_x01", "thomas_tpl_p10_x01", "thomas_tpl_p20_x01"]
fig, ax = PyPlot.subplots()
for (i, network_dir) in enumerate(network_dirs)
	resultsfilename = "results/" * replace(network_dir, "/"=>"_") * "_results.jld"
	thesetimes, thesechs = JLD.load(resultsfilename, "times", "chs")
	times[network_dir] = thesetimes
	chs[network_dir] = thesechs
end
for name in names
	if sum(map(network_dir->chs[network_dir][name][end] < 1e-10, network_dirs)) > 2
		ax.loglog(map(network_dir->numnodes[network_dir], network_dirs), map(network_dir->ifelse(chs[network_dir][name][end] < 1e-10, times[network_dir][name], NaN), network_dirs), ".-", label=name, alpha=0.75, linewidth=3, ms=10)
	end
end
ax.legend(fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_xlabel("Number of grid nodes", fontsize=16)
ax.set_ylabel("Run time (seconds)", fontsize=16)
fig.tight_layout()
fig.savefig("node_scaling.pdf")
display(fig)
println()
PyPlot.close(fig)

#do the plot on the b scaling for the homogenous dfns
network_dirs = ["var_b_0.0", "var_b_0.5", "var_b_1.0"]
bs = Dict(zip(network_dirs, ["0", "0.5", "1.0"]))
fig, ax = PyPlot.subplots()
for (i, network_dir) in enumerate(network_dirs)
	resultsfilename = "results/" * replace(network_dir, "/"=>"_") * "_results.jld"
	thesetimes, thesechs = JLD.load(resultsfilename, "times", "chs")
	times[network_dir] = thesetimes
	chs[network_dir] = thesechs
end
for name in names
	if sum(map(network_dir->chs[network_dir][name][end] < 1e-10, network_dirs)) > 2
		ax.semilogy(map(network_dir->bs[network_dir], network_dirs), map(network_dir->ifelse(chs[network_dir][name][end] < 1e-10, times[network_dir][name], NaN), network_dirs), ".-", label=name, alpha=0.75, linewidth=3, ms=10)
	end
end
ax.legend(fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_xlabel("Variance of the log-aperture", fontsize=16)
ax.set_ylabel("Run time (seconds)", fontsize=16)
fig.tight_layout()
fig.savefig("b_scaling.pdf")
display(fig)
println()
PyPlot.close(fig)
