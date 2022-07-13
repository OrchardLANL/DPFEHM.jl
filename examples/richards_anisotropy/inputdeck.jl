#This solves the richards equations for a heterogeneous permeability field with an anisotropic covariance function.
#There is a fluid source in the top middle of the domain and the medium is saturated at the bottom of the domain.
#Gravity pulls the fluid from the top toward the bottom.
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
		obsnodes[i] = findmin(map(j->sum((obslocs[i] .- coords[[1, 3], j]) .^ 2), 1:size(coords, 2)))[2]
	end
	return obsnodes
end

mins = [0, 0]#meters
maxs = [100, 10]#meters
#ns = [11, 11]
#ns = [21, 21]
#ns = [51, 51]
ns = [101, 101]
#ns = [201, 201]
#ns = [401, 401]
#ns = [801, 801]
num_eigenvectors = 200
x_true = randn(num_eigenvectors)
x0 = zeros(num_eigenvectors)
sqrtnumobs = 16
obslocs_x = range(mins[1], maxs[1]; length=sqrtnumobs + 2)[2:end - 1]
obslocs_z = range(mins[2], maxs[2]; length=sqrtnumobs + 2)[2:end - 1]
obslocs = collect(Iterators.product(obslocs_x, obslocs_z))[:]
observations = Array{Float64}(undef, length(obslocs))

	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
	coords = vcat(coords, zeros(size(coords, 2))')#add a z component
	coords[3, :] = coords[2, :]#swap the y and z components
	coords[2, :] .= 0.0
	Qs = zeros(size(coords, 2))
	boundaryhead(x, y, z) = 0.0 - z
	dirichletnodes = Int[]
	dirichleths = zeros(size(coords, 2))
	for i = 1:size(coords, 2)
		if coords[3, i] == mins[2]
			push!(dirichletnodes, i)
			dirichleths[i] = boundaryhead(coords[:, i]...)
		elseif coords[3, i] == maxs[2] && coords[1, i] > maxs[1] * 1 / 4 && coords[1, i] < maxs[1] * 3 / 4
			Qs[i] = 1e-2
		end
	end

	lambda = 10.0#meters -- correlation length of log-conductivity
	sigma = 1.0#standard deviation of log-conductivity
	mu = 0.0#mean of log-permeability can be arbitrary because this is steady-state and there are no fluid sources

	#cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
	#anisotropy = [1 / lambda 0.1 / lambda;  / lambda 1 / lambda]
	anisotropy = [1 0.9; 0.9 1] / lambda
	cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.AnisotropicExponential(anisotropy, σ=sigma))
	x_pts = range(mins[1], maxs[1]; length=ns[1])
	z_pts = range(mins[2], maxs[2]; length=ns[2])
	#@time grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, z_pts)
	@time grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, x_pts)
	@time logKs = GaussianRandomFields.sample(grf)
	alphas = fill(0.375, length(neighbors))
	Ns = fill(1.25, length(neighbors))
	parameterization = copy(grf.data.eigenfunc)
	sigmas = copy(grf.data.eigenval)

	#plot a realization
	fig, ax = PyPlot.subplots()
	ax.imshow(logKs, origin="lower")
	ax.title.set_text("Random Permeability Field")
	display(fig)
	println()
	PyPlot.close(fig)

	#plot the first few eigenvectors and some random ones
	fig, axs = PyPlot.subplots(2, 4, figsize=(16, 8))
	rp = Random.randperm(size(parameterization, 2))
	for (i, ax) in enumerate(axs)
		if i <= length(axs) / 2
			ax.imshow(reshape(parameterization[:, i], ns...), origin="lower")
			ax.title.set_text("Eigenvector $i")
		else
			ax.imshow(reshape(parameterization[:, rp[i]], ns...), origin="lower")
			ax.title.set_text("Eigenvector $(rp[i])")
		end
	end
	display(fig)
	println()
	PyPlot.close(fig)

	#logKs2Ks_neighbors(Ks) = map(neighbor->exp(0.5 * (Ks[neighbor[1]] + Ks[neighbor[2]])), neighbors)
	#logKs2Ks_neighbors(Ks) = exp.([0.5 * (Ks[i1] + Ks[i2]) for (i1, i2) in neighbors])
	logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#Zygote differentiates this efficiently but the definitions above are ineffecient with Zygote
	function solveforh(logKs, dirichleths)
		@assert length(logKs) == length(Qs)
		if maximum(logKs) - minimum(logKs) > 25
			return fill(NaN, length(Qs))#this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
		else
			Ks_neighbors = logKs2Ks_neighbors(logKs)
			psi0 = map(i->boundaryhead(coords[:, i]...), 1:size(coords, 2))
			return DPFEHM.richards_steadystate(psi0, Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, coords, alphas, Ns, Qs)
		end
	end
	solveforheigs(x) = solveforh(x2logKs(x), dirichleths)
	x2logKs(x) = reshape(parameterization * (sigmas .* x), ns...)
	logKs_true = x2logKs(x_true)
	h_true = solveforheigs(x_true)

	obsnodes = getobsnodes(coords, obslocs)
	obssigma = 1e-3
	observations .= h_true[obsnodes]#set up the observations
	f(logKs) = sum((solveforh(logKs, dirichleths)[obsnodes] - observations) .^ 2 ./ obssigma ^ 2)
	print("forward solve time")
	@time h = solveforh(logKs, dirichleths)
	print("gradient time")
	@time zg = Zygote.gradient(f, logKs)[1]

	#plot the solution, the difference between the solution and the solution without recharge, and the logKs
	fig, axs = PyPlot.subplots(1, 4, figsize=(16, 4))
	ims = axs[1].imshow(reshape(h, ns...), origin="lower")
	axs[1].title.set_text("Head for Random Permeability")
	fig.colorbar(ims, ax=axs[1])
	ims = axs[2].imshow(DPFEHM.effective_saturation.(alphas[1], reshape(h, ns...), Ns[1]), origin="lower")
	axs[2].title.set_text("Effective Saturation\nfor Random Permeability")
	fig.colorbar(ims, ax=axs[2])
	ims = axs[3].imshow(logKs, origin="lower")
	axs[3].title.set_text("Random Permeability Field")
	fig.colorbar(ims, ax=axs[3])
	ims = axs[4].imshow(zg, origin="lower")
	axs[4].title.set_text("Gradient of Loss\nw.r.t. Permeability")
	fig.colorbar(ims, ax=axs[4])
	fig.tight_layout()
	display(fig)
	println()
	PyPlot.close(fig)
