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

function groundwater_steadystate(Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; kwargs...)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	args = (zeros(sum(isfreenode)), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args...)
	A = DPFEHM.groundwater_h(args...)
	ml = AlgebraicMultigrid.ruge_stuben(A)
	hfree = AlgebraicMultigrid.solve(ml, b; kwargs...)
	h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
	return h
end

function ChainRulesCore.rrule(::typeof(groundwater_steadystate), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; kwargs...)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	args_noh = (zeros(sum(isfreenode)), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args_noh...)
	A = DPFEHM.groundwater_h(args_noh...)
	ml = AlgebraicMultigrid.ruge_stuben(A)
	hfree = AlgebraicMultigrid.solve(ml, b; kwargs...)
	h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
	function pullback(delta)
		args = (hfree, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
		ml_A_prime = AlgebraicMultigrid.ruge_stuben(SparseArrays.SparseMatrixCSC(A'))
		lambda = AlgebraicMultigrid.solve(ml_A_prime, delta[isfreenode]; kwargs...)
		gw_Ks = groundwater_Ks(args...)
		gw_dirichleths = groundwater_dirichleths(args...)
		gw_Qs = groundwater_Qs(args...)
		return (ChainRulesCore.NO_FIELDS,#step function
				@ChainRulesCore.thunk(-(gw_Ks' * lambda)),#Ks
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#neighbors
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#areasoverlengths
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#dirichletnodes
				@ChainRulesCore.thunk(-(gw_dirichleths' * lambda) .+ delta .* (map(x->!x, isfreenode))),#dirichleths
				@ChainRulesCore.thunk(-(gw_Qs' * lambda)))#Qs
	end
	return h, pullback
end
