import Zygote

module TheisLike
	#push!(LOAD_PATH,"/Users/dharp/source/DPFEHM.jl/src")
	import DifferentiableBackwardEuler
	import DPFEHM

	n = 51
	ns = [n, n]
	steadyhead = 0e0
	sidelength = 200
	thickness = 1.0
	coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d([-sidelength, -sidelength], [sidelength, sidelength], ns, thickness)
	dirichletnodes = Int[]
	dirichleths = zeros(size(coords, 2))
	for i = 1:size(coords, 2)
		if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
			push!(dirichletnodes, i)
			dirichleths[i] = steadyhead
		end
	end

	function getQs(Qs::Vector, rs::Vector)
		return sum(map(i->getQs(Qs[i], rs[i]), 1:length(Qs)))
	end

	function getQs(Q::Number, r::Number)#this splits the Q across all the nodes that are as close as possible to r (should be at least 4)
		dists = map(i->(sqrt(sum(coords[:, i] .^ 2)) - r) ^ 2, 1:size(coords, 2))
		mindist = minimum(dists)
		goodnodes = (dists .≈ mindist)
		Qs = Q * goodnodes / sum(goodnodes)
		return Qs
	end

	function solve_numerical(Qs, T, S, t, rs)
		Ks = fill(T, length(neighbors))#note the thickness is 1
		specificstorage = fill(S, size(coords, 2))#note the thickness is 1
		Qs = getQs(Qs, rs)
		h0 = fill(steadyhead, size(coords, 2))
		p = [Qs; Ks; specificstorage]
		h_gw = DifferentiableBackwardEuler.steps(h0, f_gw, f_gw_u, f_gw_p, f_gw_t, p, 0.0, t; abstol=1e-2, reltol=1e-2)
		goodnode = div(size(coords, 2), 2) + 1
		@assert coords[:, goodnode] == [0, 0]#make sure we are looking at the right "good node"
		return h_gw[goodnode, end] - steadyhead
	end

	function unpack(p)
		@assert length(p) == length(neighbors) + 2 * size(coords, 2)
		Qs, Ks, specificstorage = p[1:size(coords, 2)], p[size(coords, 2) + 1:size(coords, 2) + length(neighbors)], p[size(coords, 2) + length(neighbors) + 1:size(coords, 2) + length(neighbors) + size(coords, 2)]
		return Qs, Ks, specificstorage
	end
	function f_gw(u, p, t)
		Qs, Ks, specificstorage = unpack(p)
		return DPFEHM.groundwater_residuals(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
	end
	function f_gw_u(u, p, t)
		Qs, Ks, specificstorage = unpack(p)
		retval = DPFEHM.groundwater_h(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
		return retval
	end
	function f_gw_p(u, p, t)
		Qs, Ks, specificstorage = unpack(p)
		J1 = DPFEHM.groundwater_Qs(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
		J2 = DPFEHM.groundwater_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
		J3 = DPFEHM.groundwater_specificstorage(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
		return hcat(J1, J2, J3)
	end
	f_gw_t(u, p, t) = zeros(length(u))
end

#Qs = [0.031688, -0.031688]
#T = 0.1
#S = 0.01
#t = 60 * 60 * 24.0 * 30
#rs = [sqrt(2) * 125, sqrt(2) * 25]
#@time @show TheisLike.solve_numerical(Qs, T, S, t, rs)
#@time gradient = Zygote.gradient(Qs->TheisLike.solve_numerical(Qs, T, S, t, rs), Qs)[1]
#@show gradient
