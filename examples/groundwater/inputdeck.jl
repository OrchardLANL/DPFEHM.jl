import DPFEHM
import GaussianRandomFields
import Optim
import PyPlot
import Random
import Zygote

Random.seed!(0)

function getobsnodes(coords, obslocs)
	obsnodes = Array{Int}(undef, length(obslocs))
	for i = 1:length(obslocs)
		obsnodes[i] = findmin(map(j->sum((obslocs[i] .- coords[:, j]) .^ 2), 1:size(coords, 2)))[2]
	end
	return obsnodes
end

mins = [0, 0]#meters
maxs = [100, 100]#meters
#allns = [[101, 101], [201, 201], [401, 401], [801, 801], [1601, 1601]]
allns = [[101, 101], [201, 201], [401, 401]]
allns = [[allns[end]]; allns]
num_eigenvectors = 200
x_true = randn(num_eigenvectors)
x0 = zeros(num_eigenvectors)
sqrtnumobs = 16
obslocs_x = range(mins[1], maxs[1]; length=sqrtnumobs + 2)[2:end - 1]
obslocs_y = range(mins[2], maxs[2]; length=sqrtnumobs + 2)[2:end - 1]
obslocs = collect(Iterators.product(obslocs_x, obslocs_y))[:]
observations = Array{Float64}(undef, length(obslocs))
for (i, ns) in enumerate(allns)
	Random.seed!(0)
	global x_true
	global x0
	global observations
	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
	Qs = zeros(size(coords, 2))
	boundaryhead(x, y) = 5 * (x - maxs[1]) / (mins[1] - maxs[1])
	dirichletnodes = Int[]
	dirichleths = zeros(size(coords, 2))
	for i = 1:size(coords, 2)
		if coords[1, i] == mins[1] || coords[1, i] == maxs[1] || coords[2, i] == mins[2] || coords[2, i] == maxs[2]
			push!(dirichletnodes, i)
			dirichleths[i] = boundaryhead(coords[1:2, i]...)
		end
	end

	lambda = 50.0#meters -- correlation length of log-conductivity
	sigma = 1.0#standard deviation of log-conductivity
	mu = 0.0#mean of log-permeability can be arbitrary because this is steady-state and there are no fluid sources

	cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
	x_pts = range(mins[1], maxs[1]; length=ns[1])
	y_pts = range(mins[2], maxs[2]; length=ns[2])
	@time grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
	@time logKs = GaussianRandomFields.sample(grf)
	parameterization = copy(grf.data.eigenfunc)
	sigmas = copy(grf.data.eigenval)

	if i == 1#only plot this the first time around
		#plot a realization
		fig, ax = PyPlot.subplots()
		ax.imshow(logKs)
		ax.title.set_text("Random Conductivity Field")
		display(fig)
		println()
		PyPlot.close(fig)

		#plot the first few eigenvectors and some random ones
		fig, axs = PyPlot.subplots(2, 4, figsize=(16, 8))
		rp = Random.randperm(size(parameterization, 2))
		for (i, ax) in enumerate(axs)
			if i <= length(axs) / 2
				ax.imshow(reshape(parameterization[:, i], ns...))
				ax.title.set_text("Eigenvector $i")
			else
				ax.imshow(reshape(parameterization[:, rp[i]], ns...))
				ax.title.set_text("Eigenvector $(rp[i])")
			end
		end
		display(fig)
		println()
		PyPlot.close(fig)
	end

	#logKs2Ks_neighbors(Ks) = map(neighbor->exp(0.5 * (Ks[neighbor[1]] + Ks[neighbor[2]])), neighbors)
	#logKs2Ks_neighbors(Ks) = exp.([0.5 * (Ks[i1] + Ks[i2]) for (i1, i2) in neighbors])
	logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#Zygote differentiates this efficiently but the definitions above are ineffecient with Zygote
	function solveforh(logKs, dirichleths)
		@assert length(logKs) == length(Qs)
		if maximum(logKs) - minimum(logKs) > 25
			return fill(NaN, length(Qs))#this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
		else
			Ks_neighbors = logKs2Ks_neighbors(logKs)
			return DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
		end
	end
	solveforheigs(x) = solveforh(x2logKs(x), dirichleths)
	x2logKs(x) = reshape(parameterization * (sigmas .* x), ns...)
	logKs_true = x2logKs(x_true)
	h_true = solveforheigs(x_true)

	numobs = 250
	obsnodes = getobsnodes(coords, obslocs)
	obssigma = 1e-3
	observations .= h_true[obsnodes]#set up the observations
	f(logKs) = sum((solveforh(logKs, dirichleths)[obsnodes] - observations) .^ 2 ./ obssigma ^ 2)
	h = solveforh(logKs, dirichleths)
	h = solveforh(logKs, dirichleths)
	print("forward solve time")
	@time h = solveforh(logKs_true, dirichleths)
	zg = Zygote.gradient(f, logKs)[1]
	zg = Zygote.gradient(f, logKs)[1]#work out the precompilation
	print("gradient time")
	@time zg = Zygote.gradient(f, logKs)[1]

	#now reformulate the function in terms of the cofficients of the eigenvectors with some regularization in there
	feigs(x) = f(x2logKs(x)) + sum(x .^ 2)
	print("eigs forward solve time")
	@time feigs(x_true)
	zgeigs = Zygote.gradient(feigs, x_true)[1]
	print("eigs gradient time")
	@time zgeigs = Zygote.gradient(feigs, x_true)[1]

	if i == 1#first time around plot a bunch of stuff and set up the observations
		#plot the solution, the difference between the solution and the solution for a uniform medium, and the logKs
		fig, axs = PyPlot.subplots(1, 4, figsize=(32, 8))
		ims = axs[1].imshow(reshape(h, ns...))
		axs[1].title.set_text("Head for Random Conductivity")
		fig.colorbar(ims, ax=axs[1])
		ims = axs[2].imshow(reshape(h .- map(i->boundaryhead(coords[:, i]...), 1:size(coords, 2)), ns...))
		axs[2].title.set_text("Head Fluctuation for Random Conductivity")
		fig.colorbar(ims, ax=axs[2])
		ims = axs[3].imshow(logKs)
		axs[3].title.set_text("Random Conductivity Field")
		fig.colorbar(ims, ax=axs[3])
		ims = axs[4].imshow(zg)
		axs[4].title.set_text("Gradient of Loss w.r.t. Conductivity")
		fig.colorbar(ims, ax=axs[4])
		fig.tight_layout()
		display(fig)
		println()
		PyPlot.close(fig)

		#plot the solution, the difference between the solution and the solution for a uniform medium, and the logKs
		fig, axs = PyPlot.subplots(1, 4, figsize=(32, 8))
		ims = axs[1].imshow(reshape(h_true, ns...))
		axs[1].title.set_text("Head for Random Conductivity")
		fig.colorbar(ims, ax=axs[1])
		ims = axs[2].imshow(reshape(h_true .- map(i->boundaryhead(coords[:, i]...), 1:size(coords, 2)), ns...))
		axs[2].title.set_text("Head Fluctuation for Random Conductivity")
		fig.colorbar(ims, ax=axs[2])
		ims = axs[3].imshow(x2logKs(x_true))
		axs[3].title.set_text("True Conductivity Field")
		fig.colorbar(ims, ax=axs[3])
		ims = axs[4].plot(zgeigs)
		axs[4].title.set_text("Gradient of Loss w.r.t. Eigenvector Coefficients")
		fig.tight_layout()
		display(fig)
		println()
		PyPlot.close(fig)

	else#otherwise, do the optimization
		options = Optim.Options(iterations=200, extended_trace=false, store_trace=true, show_trace=ifelse(i == length(allns), true, false), x_tol=1e-6, time_limit=60 * 60 * 2)
		print("Optimization time")
		@time opt = Optim.optimize(feigs, x->Zygote.gradient(feigs, x)[1], x0, Optim.LBFGS(), options; inplace=false)
		#@time opt = Optim.optimize(feigs, x->Zygote.gradient(feigs, x)[1], x0, Optim.GradientDescent(), options; inplace=false)
		x_est = opt.minimizer
		logKs_est = x2logKs(x_est)

		#plot results from the inverse analysis
		fig, axs = PyPlot.subplots(2, 3, figsize=(12, 8))
		ims = axs[1].imshow(logKs_true)
		axs[1].title.set_text("True Conductivity")
		fig.colorbar(ims, ax=axs[1])
		ims = axs[2].imshow(logKs_est)
		axs[2].title.set_text("Estimated Conductivity")
		fig.colorbar(ims, ax=axs[2])
		ims = axs[3].imshow(reshape(solveforh(logKs_true, dirichleths), ns...))
		axs[3].title.set_text("True Head")
		#ims = axs[3].imshow(x2logKs(x0))
		#axs[3].title.set_text("Previous Conductivity Estimate")
		fig.colorbar(ims, ax=axs[3])
		#ims = axs[4].imshow(logKs_est - x2logKs(x0))
		#axs[4].title.set_text("Conductivity Update")
		ims = axs[4].imshow(reshape(solveforh(logKs_est, dirichleths), ns...))
		axs[4].title.set_text("Estimated Head")
		fig.colorbar(ims, ax=axs[4])
		axs[5].plot([0, 5], [0, 5], "k", alpha=0.5)
		axs[5].plot(observations, solveforh(logKs_est, dirichleths)[obsnodes], "k.")
		axs[5].set_xlabel("Observed Head")
		axs[5].set_ylabel("Predicted Head")
		axs[6].semilogy(map(i->opt.trace[i].value, 1:length(opt.trace)))
		axs[6].set_xlabel("Iteration")
		axs[6].set_ylabel("Loss")
		fig.tight_layout()
		fig.savefig("figs/level_$(i - 1).pdf")
		display(fig)
		println()
		PyPlot.close(fig)

		x0 = x_est#update x0 for the next optimization
	end
end
