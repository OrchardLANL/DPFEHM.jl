ForwardDiff_gradient(x...) = ForwardDiff.gradient(x...)

function kr(psi, alpha, N)
	if psi < 0
		m = (N - 1) / N
		denom = 1 + abs(alpha * psi) ^ N
		numer = 1 - abs(alpha * psi) ^ (N - 1) * denom ^ (-m)
		return numer ^ 2 / denom ^ (m / 2)
	else
		return one(psi)
	end
end

kr(x::AbstractArray) = kr(x[1], x[2], x[3])

function Calculus.differentiate(x::Calculus.SymbolParameter{:kr}, args, wrt)
	chain_part = map(x->Calculus.simplify(Calculus.differentiate(x, wrt)), args)
	if chain_part == [0, 0, 0]
		return :(0)
	else
		return :(sum(ForwardDiff_gradient(kr, StaticArrays.SA[$(args...)]) .* StaticArrays.SA[$(chain_part...)]))
	end
end

function hm(x, y)
	return 2 / (1 / x + 1 / y)
end

hm(x::AbstractArray) = hm(x[1], x[2])

function Calculus.differentiate(x::Calculus.SymbolParameter{:hm}, args, wrt)
	chain_part = map(x->Calculus.simplify(Calculus.differentiate(x, wrt)), args)
	if chain_part == [0, 0]
		return :(0)
	else
		return :(sum(ForwardDiff_gradient(hm, StaticArrays.SA[$(args...)]) .* StaticArrays.SA[$(chain_part...)]))
	end
end

function Calculus.differentiate(x::Calculus.SymbolParameter{:abs}, args, wrt)
	if length(args) > 1
		error("Too many arguments passed to abs()")
	end
	arg = args[1]
	return :(ifelse($arg > 0, 1, -1) * $(Calculus.differentiate(arg, wrt)))
end

function effective_saturation(alpha::Number, psi::Number, N::Number)
	m = (N - 1) / N
	if psi < 0
		return (1 + abs(alpha * psi) ^ N) ^ (-m)
	else
		return 1
	end
end

@NonlinearEquations.equations exclude=(coords, dirichletnodes, neighbors, areasoverlengths) function richards(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
	NonlinearEquations.setnumequations(length(psi))
	dirichletnodes_set = Set(dirichletnodes)
	for i = 1:length(psi)
		if i in dirichletnodes_set
			NonlinearEquations.addterm(i, psi[i] - dirichletpsis[i])
		else
			NonlinearEquations.addterm(i, Qs[i] / (specificstorage[i] * volumes[i]))
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes_set) && !(node2 in dirichletnodes_set)
				NonlinearEquations.addterm(node1, hm(kr(psi[node1], alphas[i], Ns[i]), kr(psi[node2], alphas[i], Ns[i])) * Ks[i] * (psi[node2] + coords[3, node2] - psi[node1] - coords[3, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			elseif !(node1 in dirichletnodes_set) && node2 in dirichletnodes_set
				NonlinearEquations.addterm(node1, hm(kr(psi[node1], alphas[i], Ns[i]), kr(dirichletpsis[node2], alphas[i], Ns[i])) * Ks[i] * (dirichletpsis[node2] + coords[3, node2] - psi[node1] - coords[3, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			end
		end
	end
end

function richards_steadystate(psi0, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; kwargs...)
	args = (psi0, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, ones(length(Qs)), ones(length(Qs)))
	function residuals!(residuals, psi)
		myargs = (psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, ones(length(Qs)), ones(length(Qs)))
		copy!(residuals, DPFEHM.richards_residuals(myargs...))
	end
	function jacobian!(J, psi)
		myargs = (psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, ones(length(Qs)), ones(length(Qs)))
		DPFEHM.richards_psi!(J, myargs...)
	end
	J0 = DPFEHM.richards_psi(args...)
	residuals0 = DPFEHM.richards_residuals(args...)
	df = NLsolve.OnceDifferentiable(residuals!, jacobian!, psi0, residuals0, J0)
	soln = NLsolve.nlsolve(df, psi0; kwargs...)
	if !NLsolve.converged(soln)
		display(soln)
		error("solution did not converge")
	end
	return soln.zero
end

function ChainRulesCore.rrule(::typeof(richards_steadystate), psi0, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; kwargs...)
	psi = richards_steadystate(psi0, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; kwargs...)
	function pullback(delta)
		args = (psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, ones(length(Qs)), ones(length(Qs)))
		lambda = richards_psi(args...)' \ delta
		req_Ks = richards_Ks(args...)
		req_dirichletpsis = richards_dirichletpsis(args...)
		req_alphas = richards_alphas(args...)
		req_Ns = richards_Ns(args...)
		req_Qs = richards_Qs(args...)
		return (ChainRulesCore.NO_FIELDS,#step function
				@ChainRulesCore.thunk(zeros(length(psi0))),#psi0 -- should be insensitive to psi0
				@ChainRulesCore.thunk(-(req_Ks' * lambda)),#Ks
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#neighbors
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#areasoverlengths
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#dirichletnodes
				@ChainRulesCore.thunk(-(req_dirichletpsis' * lambda)),#dirichletpsis
				@ChainRulesCore.thunk(ChainRulesCore.NO_FIELDS),#coords
				@ChainRulesCore.thunk(-(req_alphas' * lambda)),#alphas
				@ChainRulesCore.thunk(-(req_Ns' * lambda)),#Ns
				@ChainRulesCore.thunk(-(req_Qs' * lambda)))#Qs
	end
	return psi, pullback
end
