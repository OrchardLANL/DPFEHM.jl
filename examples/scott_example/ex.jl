import DPFEHM
import GaussianRandomFields
import Optim
import PyPlot
import Random
import Zygote

function inversemodel(sigma,lambda,mu,deltah,sqrtnumobs,alpha,beta,gamma,iterations)
	#sigma is the standard deviation of logK
	#lambda is the correlation length
	#mu is the mean of logK
	#deltah is the pressure drop from the left boundary to the right boundary
	#sqrtnumobs is the square root of the number of observation wells (which are distributed on a regular sqrtnumobs-by-sqrtnumobs grid)
	#alpha is the weight associated with head measurements
	#beta is the weight associated with the logK measurements
	#gamma is the weight associated with the darcy velocity measurements
	#iterations is the maximum number of iterations that the inverse analysis will perform
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
	allns = [[101, 101]]
	allns = [[allns[end]]; allns]
	num_eigenvectors = 200
	x0 = zeros(num_eigenvectors)
	num_runs = 10

	obslocs_x = range(mins[1], maxs[1]; length=sqrtnumobs + 2)[2:end - 1]
	obslocs_y = range(mins[2], maxs[2]; length=sqrtnumobs + 2)[2:end - 1]
	obslocs = collect(Iterators.product(obslocs_x, obslocs_y))[:]
	h_obser = Array{Float64}(undef, length(obslocs))

	obslocs_x2 = range(mins[1], maxs[1]; length=sqrtnumobs + 2)[2:end - 1]
	obslocs_y2 = range(mins[1], maxs[1]; length=sqrtnumobs + 2)[2:end - 1]
	obslocs2 = collect(Iterators.product(obslocs_x2, obslocs_y2))[:]
	K_obser = Array{Float64}(undef, length(obslocs2))

	runs = collect(1:num_runs)
	M = zeros(num_eigenvectors,num_runs) #RMS of eigenvalues 
	N = zeros(iterations+1,num_runs) #RMS of logK
	P = zeros(iterations+1,num_runs) #RMS of Loss

	Random.seed!(0) 
	ns=[101,101]
	
	cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
	x_pts = range(mins[1], maxs[1]; length=ns[1])
	y_pts = range(mins[2], maxs[2]; length=ns[2])
	@time grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
	@time logKs = GaussianRandomFields.sample(grf)
	parameterization = copy(grf.data.eigenfunc)
	sigmas = copy(grf.data.eigenval)

	for n in 1:num_runs
		x_true = randn(num_eigenvectors)
	
		for (i, ns) in enumerate(allns)
	
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
			obsnodes2 = getobsnodes(coords, obslocs2)
			h_obssigma = 1e-3
			K_obssigma = 1e-1
	
			function f(logKs)#Loss function
				h_error = sum((solveforh(logKs, dirichleths)[obsnodes] - h_obser) .^ 2 ./ h_obssigma ^ 2)
				K_error = sum((logKs[obsnodes2] - K_obser) .^ 2 ./ K_obssigma ^ 2)
				error_sys = alpha*h_error + beta*K_error
				return error_sys
			end 
		
			print("forward solve time")
			@time h = solveforh(logKs_true, dirichleths)
			zg = Zygote.gradient(f, logKs)[1]#work out the precompilation
			print("gradient time")
			@time zg = Zygote.gradient(f, logKs)[1]
	
			print(' ')
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
		
				h_obser .= h_true[obsnodes]
				K_obser .= logKs_true[obsnodes2]
			else#otherwise, do the optimization
				options = Optim.Options(iterations=iterations, store_trace=true, show_trace=false, extended_trace=true, x_tol=1e-6, time_limit=60 * 60 * 2)
				print("Optimization time")
				opt = Optim.optimize(feigs, x->Zygote.gradient(feigs, x)[1], x0, Optim.LBFGS(), options; inplace=false)
										   
				N[1:length(opt.trace),n] = map(j->(sqrt.((sum((x2logKs(opt.trace[j].metadata["x"])-x2logKs(x_true)).^2))/num_eigenvectors)),1:length(opt.trace))
				P[1:length(opt.trace),n] = map(j->(sqrt.(opt.trace[j].value)), 1:length(opt.trace))
			
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
				fig.colorbar(ims, ax=axs[3])
				ims = axs[4].imshow(reshape(solveforh(logKs_est, dirichleths), ns...))
				axs[4].title.set_text("Estimated Head")
				fig.colorbar(ims, ax=axs[4])
				axs[5].plot([0, 5], [0, 5], "k", alpha=0.5)
				axs[5].plot(h_obser, solveforh(logKs_est, dirichleths)[obsnodes], "k.")
				axs[5].set_xlabel("Observed Head")
				axs[5].set_ylabel("Predicted Head")
				axs[6].semilogy(map(i->opt.trace[i].value, 1:length(opt.trace)))
				axs[6].set_xlabel("Iteration")
				axs[6].set_ylabel("Loss")
				fig.tight_layout()
				display(fig)
				println()
				PyPlot.close(fig)
	
				x0 = x_est#update x0 for the next optimization
				M[:,n] = x_est-x_true
			end
		end
		println("$n")
	end 
		
	println("done!")

	E = zeros(200)
	for i in 1:num_eigenvectors
		E[i] = sqrt.(sum(M[i,:].^2)/num_runs)
	end
	
	fig, axs = PyPlot.subplots(1, 2, figsize=(10, 4))
	axs[1].semilogy(E,".")
	axs[1].set_xlabel("Eigenvectors numbers")
	axs[1].set_ylabel("RMS of coeffient error")
	axs[2].loglog(P,N,".")
	axs[2].set_xlabel("RMS of Loss")
	axs[2].set_ylabel("RMS of logKs")
end

inversemodel(1.0,50.0,0.0,5.0,16,0.2,0.8,0.5,200)
