using Distributed
@everywhere begin
	import DPFEHM
	import GaussianRandomFields
	import Optim
	import PyPlot
	import Random
	import Zygote

	function inversemodel(sigma, lambda, mu, deltah, sqrtnumobs, alpha, beta, gamma, iterations)
		#sigma is the standard deviation of logK
		#lambda is the correlation length
		#mu is the mean of logK
		#deltah is the pressure drop from the left boundary to the right boundary
		#sqrtnumobs is the square root of the number of observation wells (which are distributed on a regular sqrtnumobs-by-sqrtnumobs grid)
		#alpha is the weight associated with head measurements
		#beta is the weight associated with the logK measurements
		#gamma is the weight associated with the darcy velocity measurements
		#iterations is the maximum number of iterations that the inverse analysis will perform

		function getobsnodes(coords, obslocs)
			obsnodes = Array{Int}(undef, length(obslocs))
			for i = 1:length(obslocs)
				obsnodes[i] = findmin(map(j->sum((obslocs[i] .- coords[:, j]) .^ 2), 1:size(coords, 2)))[2]
			end
			return obsnodes
		end

		mins = [0.0, 0.0]#meters
		maxs = [100.0, 100.0]#meters
		ns = [101, 101]
		dx = (maxs[1] - mins[1]) / (ns[1] - 1)
		dy = (maxs[2] - mins[2]) / (ns[2] - 1)
		num_eigenvectors = 200
		x0 = zeros(num_eigenvectors)

		eigenvalue_error = fill(NaN, num_eigenvectors) #RMS of eigenvalues 
		logK_error = fill(NaN, iterations + 1) #RMS of logK
		obj_func_values = fill(NaN, iterations + 1) #RMS of Loss
		
		#set up the eigenvector parameterization of the geostatistical field
		cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
		x_pts = range(mins[1], maxs[1]; length=ns[1])
		y_pts = range(mins[2], maxs[2]; length=ns[2])
		grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
		logKs = GaussianRandomFields.sample(grf)
		parameterization = copy(grf.data.eigenfunc)
		sigmas = copy(grf.data.eigenval)

		#set up the mesh  and boundary conditions
		coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
		Qs = zeros(size(coords, 2))
		boundaryhead(x, y) = deltah * (x - maxs[1]) / (mins[1] - maxs[1])
		dirichletnodes = Int[]
		dirichleths = zeros(size(coords, 2))
		for i = 1:size(coords, 2)
			if coords[1, i] == mins[1] || coords[1, i] == maxs[1] || coords[2, i] == mins[2] || coords[2, i] == maxs[2]
				push!(dirichletnodes, i)
				dirichleths[i] = boundaryhead(coords[1:2, i]...)
			end
		end

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

		#set up the observations
		obslocs_x = range(mins[1], maxs[1]; length=sqrtnumobs + 2)[2:end - 1]
		obslocs_y = range(mins[2], maxs[2]; length=sqrtnumobs + 2)[2:end - 1]
		obslocs = collect(Iterators.product(obslocs_x, obslocs_y))[:]
		x_true = randn(num_eigenvectors)
		logKs_true = x2logKs(x_true)
		h_true = solveforheigs(x_true) 
		obsnodes = getobsnodes(coords, obslocs)
		h_obser = h_true[obsnodes]
		logK_obser = logKs_true[obsnodes]
		hmat = reshape(h_true, reverse(ns)...)
		Ks = exp.(logKs)
		vy = Ks .* vcat(fill(0.0, ns[1])', hmat[3:end, :] - hmat[1:end - 2, :], fill(0.0, ns[1])') / (2 * dy)
		vx = Ks .* hcat(fill(0.0, ns[2]), hmat[:, 3:end] - hmat[:, 1:end - 2], fill(0.0, ns[2])) / (2 * dx)
		vel_obser = map(i->sqrt(vx[i] ^ 2 + vy[i] ^ 2), obsnodes)

		#set up the loss function
		function f(logKs)
			h_obssigma = 1e-3
			logK_obssigma = 1e-1
			vel_obssigma = 1e-1
			h = solveforh(logKs, dirichleths)
			h_error = sum((h[obsnodes] - h_obser) .^ 2 ./ h_obssigma ^ 2)
			K_error = sum((logKs[obsnodes] - logK_obser) .^ 2 ./ logK_obssigma ^ 2)
			Ks = exp.(logKs)
			hmat = reshape(h, reverse(ns)...)
			vy = Ks .* vcat(fill(0.0, ns[1])', hmat[3:end, :] - hmat[1:end - 2, :], fill(0.0, ns[1])') / (2 * dy)
			vx = Ks .* hcat(fill(0.0, ns[2]), hmat[:, 3:end] - hmat[:, 1:end - 2], fill(0.0, ns[2])) / (2 * dx)
			vel_pred = map(i->sqrt(vx[i] ^ 2 + vy[i] ^ 2), obsnodes)
			vel_error = sum((vel_pred - vel_obser) .^ 2 ./ vel_obssigma ^ 2)
			error_sys = alpha * h_error + beta * K_error + gamma * vel_error
			return error_sys
		end 
		feigs(x) = f(x2logKs(x)) + sum(x .^ 2)
		
		#solve the inverse problem
		options = Optim.Options(iterations=iterations, store_trace=true, show_trace=false, extended_trace=true, x_tol=1e-6, time_limit=60 * 60 * 2)
		opt = Optim.optimize(feigs, x->Zygote.gradient(feigs, x)[1], x0, Optim.LBFGS(), options; inplace=false)
		logK_error[1:length(opt.trace)] = map(j->(sqrt.((sum((x2logKs(opt.trace[j].metadata["x"])-x2logKs(x_true)).^2))/num_eigenvectors)),1:length(opt.trace))
		obj_func_values[1:length(opt.trace)] = map(j->(sqrt.(opt.trace[j].value)), 1:length(opt.trace))
		x_est = opt.minimizer
		eigenvalue_error .= x_est - x_true
		return eigenvalue_error, logK_error, obj_func_values
	end
end#end the distributed portion
