import DelimitedFiles
import DPFEHM
import PyPlot
import Random
Random.seed!(7)#this seed makes it so the wells aren't on the boundary

function addfractures!(logks, fracture_logk, fracture_scales; beta=0.5)
	#this recursively defines a formula where fracture_logk ∝ length ^ beta as in equation 5 from Jeffrey's paper (10.1002/2016WR018806)
	if fracture_scales > 0
		if fracture_logk < 0.0
			@show fracture_logk
			@show fracture_scales
			error("you need to increase fracture_logk so that the permeability of the small fractures is greater than the permeability of the matrix")
		end
		logks[div(end, 2), 1:div(3 * end, 4) + 1] .= fracture_logk + log(1.5 ^ beta)#this fracture is 1.5 times longer than the next fracture
		logks[div(end, 4):div(3 * end, 4), div(end, 2)] .= fracture_logk
		n = size(logks, 1)
		#addfractures!(view(logks, 1:div(n, 2), 1:div(n, 2)))#upper left
		addfractures!(view(logks, 1:div(n, 2), div(n, 2) + 1:n), fracture_logk - log(2 ^ beta), fracture_scales - 1; beta=beta)#upper right
		#addfractures!(view(logks, div(n, 2) + 1:n, 1:div(n, 2)))#lower left
		addfractures!(view(logks, div(n, 2) + 1:n, div(n, 2) + 1:n), fracture_logk - log(2 ^ beta), fracture_scales - 1; beta=beta)#lower right
	end
end
addboundaries(logks) = hcat(logks[:, 1], logks, logks[:, end])

function fractal_fractures(N, fracture_scales; doplot=false, matrix_logk=0.0, fracture_logk=1.0, dirichlet=true, neumann=false, caging=false, num_random_nodes=1, kwargs...)
	mins = [0.0, 0.0]
	maxs = [1, 1]
	ns = [N + 2, N]
	xs = range(mins[1], maxs[1]; length=ns[1])
	ys = range(mins[2], maxs[2]; length=ns[2])
	logks = fill(matrix_logk, N, N)
	addfractures!(logks, fracture_logk, fracture_scales; kwargs...)
	logks = addboundaries(logks)
	#make the permeability at the two leftmost columns be the matrix -- this makes it so the dirichlet boundary condition corresponds to the simple Hadamard thing
	logks[:, 1] .= matrix_logk
	logks[:, 2] .= matrix_logk
	neumann_node = 2 * N + div(N, 2)#inject into the tip of the fracture
	caging_nodes = [div(3 * N, 4) * N + div(N, 4), div(3 * N, 4) * N + div(3 * N, 4)]
	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
	random_nodes = rand(1:size(coords, 2), num_random_nodes)
	if doplot
		fig, ax = PyPlot.subplots()
		img = ax.imshow(logks, extent=[mins[1], maxs[1], mins[2], maxs[2]], origin="lower")
		fig.colorbar(img)
		if neumann
			@show coords[:, neumann_node]
			ax.scatter([coords[1, neumann_node]], [coords[2, neumann_node]], c="r", s=200, alpha=1)
		elseif caging
			@show coords[:, neumann_node]
			ax.scatter([[coords[1, neumann_node]]; coords[1, caging_nodes]], [[coords[2, neumann_node]]; coords[2, caging_nodes]], c="r", s=200, alpha=1)
		end
		if num_random_nodes > 0
			ax.scatter(coords[1, random_nodes], coords[2, random_nodes], c="r", s=200, alpha=1)
		end
		display(fig)
		println()
		PyPlot.close(fig)
	end
	logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
	Ks_neighbors = logKs2Ks_neighbors(logks)
	dirichletnodes = [collect(1:N); collect(prod(ns) - N + 1:prod(ns))]
	dirichleths = zeros(size(coords, 2))
	if dirichlet
		dirichleths[1:N] .= 1.0
	end
	Qs = zeros(size(coords, 2))
	if neumann
		Qs[neumann_node] = 1
	elseif caging
		Qs[neumann_node] = 1
		Qs[caging_nodes] .= -1 / length(caging_nodes)
	end
	if num_random_nodes > 0
		Qs[random_nodes] = randn(num_random_nodes)
	end
	A = DPFEHM.groundwater_h(zeros(size(coords, 2) - length(dirichletnodes)), Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(size(coords, 2)), ones(size(coords, 2)))
	b = -DPFEHM.groundwater_residuals(zeros(size(coords, 2) - length(dirichletnodes)), Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(size(coords, 2)), ones(size(coords, 2)))
	x = A \ b
	isfreenode, nodei2freenodei, freenodei2nodei = DPFEHM.getfreenodes(prod(ns), dirichletnodes)
	h = DPFEHM.addboundaryconditions(x, dirichletnodes, dirichleths, isfreenode, nodei2freenodei)

	if doplot
		fig, axs = PyPlot.subplots(1, 2)
		axs[1].imshow(reshape(h, reverse(ns)...), origin="lower", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		axs[2].imshow(reshape(h, reverse(ns)...) .- [1 - x for x in xs, y in ys]', origin="lower", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		display(fig)
		println()
		PyPlot.close(fig)

		fig, ax = PyPlot.subplots()
		nx(n) = 0.5 * (coords[1, n[1]] + coords[1, n[2]])
		ny(n) = 0.5 * (coords[2, n[1]] + coords[2, n[2]])
		ax.scatter(map(nx, neighbors), map(ny, neighbors), c=Ks_neighbors, s=30, alpha=0.25)
		display(fig)
		println()
		PyPlot.close(fig)
	end
	return A, x, b
end

#A, x, b = fractal_fractures(2 ^ 5, 4; fracture_logk=5.0, doplot=true)
#A, x, b = fractal_fractures(2 ^ 4, 3; fracture_logk=5.0, doplot=true)
#A, x, b = fractal_fractures(2 ^ 3, 2; fracture_logk=5.0, doplot=true, dirichlet=true)
#DelimitedFiles.writedlm("b_small.csv", b, ',')
A, x, b = fractal_fractures(2 ^ 2, 1; fracture_logk=5.0, doplot=true, dirichlet=false, num_random_nodes=2)
DelimitedFiles.writedlm("b_small.csv", b, ',')
