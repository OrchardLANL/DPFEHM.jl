@NonlinearEquations.equations exclude=(dirichletnodes, neighbors, areasoverlengths) function groundwater(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	NonlinearEquations.setnumequations(sum(isfreenode))
	for i = 1:length(Qs)
		if isfreenode[i]
			j = nodei2freenodei[i]
			NonlinearEquations.addterm(j, -Qs[i] / (specificstorage[i] * volumes[i]))
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			j1 = nodei2freenodei[node1]
			if isfreenode[node1] && isfreenode[node2]
				j2 = nodei2freenodei[node2]
				NonlinearEquations.addterm(j1, Ks[i] * (h[j1] - h[j2]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			elseif isfreenode[node1] && !isfreenode[node2]
				NonlinearEquations.addterm(j1, Ks[i] * (h[j1] - dirichleths[node2]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			end
		end
	end
end

function amg_solver(A, b; kwargs...)
	ml = AlgebraicMultigrid.ruge_stuben(A)
	hfree = AlgebraicMultigrid._solve(ml, b; kwargs...)
	return hfree
end

function cholesky_solver(A, b; kwargs...)
	Af = LinearAlgebra.cholesky(A)
	hfree = Af \ b
	return hfree
end

function groundwater_steadystate(Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; linear_solver::Function=amg_solver, kwargs...)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	args = (zeros(sum(isfreenode)), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args...)
	A = DPFEHM.groundwater_h(args...)
	hfree = linear_solver(A, b; kwargs...)
	h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
	return h
end

function ChainRulesCore.rrule(::typeof(groundwater_steadystate), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; kwargs...)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	args_noh = (zeros(sum(isfreenode)), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args_noh...)
	A = DPFEHM.groundwater_h(args_noh...)
	ml = AlgebraicMultigrid.ruge_stuben(A)
	hfree = AlgebraicMultigrid._solve(ml, b; kwargs...)
	h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
	function pullback(delta)
		args = (hfree, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
		ml_A_prime = AlgebraicMultigrid.ruge_stuben(SparseArrays.SparseMatrixCSC(A'))
		lambda = AlgebraicMultigrid._solve(ml_A_prime, delta[isfreenode]; kwargs...)
		gw_Ks = groundwater_Ks(args...)
		gw_dirichleths = groundwater_dirichleths(args...)
		gw_Qs = groundwater_Qs(args...)
		return (ChainRulesCore.NoTangent(),#step function
				@ChainRulesCore.thunk(-(gw_Ks' * lambda)),#Ks
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#neighbors
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#areasoverlengths
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#dirichletnodes
				@ChainRulesCore.thunk(-(gw_dirichleths' * lambda) .+ delta .* (map(x->!x, isfreenode))),#dirichleths
				@ChainRulesCore.thunk(-(gw_Qs' * lambda)))#Qs
	end
	return h, pullback
end
