@NonlinearEquations.equations exclude=(dirichletnodes, neighbors, areasoverlengths) function groundwater(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
	NonlinearEquations.setnumequations(length(h))
	for i = 1:length(h)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, h[i] - dirichleths[i])
		else
			NonlinearEquations.addterm(i, Qs[i] / (specificstorage[i] * volumes[i]))
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, Ks[i] * (h[node2] - h[node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, Ks[i] * (dirichleths[node2] - h[node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			end
		end
	end
end

function groundwater_steadystate(Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; kwargs...)
	args = (zeros(length(Qs)), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args...)
	A = DPFEHM.groundwater_h(args...)
	ml = AlgebraicMultigrid.ruge_stuben(A)
	return AlgebraicMultigrid.solve(ml, b; kwargs...)
end

function ChainRulesCore.rrule(::typeof(groundwater_steadystate), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; kwargs...)
	args_noh = (zeros(length(Qs)), Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args_noh...)
	A = DPFEHM.groundwater_h(args_noh...)
	ml = AlgebraicMultigrid.ruge_stuben(A)
	h = AlgebraicMultigrid.solve(ml, b; kwargs...)
	function pullback(delta)
		args = (h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, ones(length(Qs)), ones(length(Qs)))
		ml_A_prime = AlgebraicMultigrid.ruge_stuben(SparseArrays.SparseMatrixCSC(A'))
		lambda = AlgebraicMultigrid.solve(ml_A_prime, delta)
		gw_Ks = groundwater_Ks(args...)
		gw_dirichleths = groundwater_dirichleths(args...)
		gw_Qs = groundwater_Qs(args...)
		return (ChainRulesCore.NO_FIELDS,#step function
				@ChainRulesCore.thunk(-(gw_Ks' * lambda)),#Ks
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#neighbors
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#areasoverlengths
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#dirichletnodes
				@ChainRulesCore.thunk(-(gw_dirichleths' * lambda)),#dirichleths
				@ChainRulesCore.thunk(-(gw_Qs' * lambda)))#Qs
	end
	return h, pullback
end
