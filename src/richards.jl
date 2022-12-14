"""
`ForwardDiff_gradient(x...)`

Wrapper for `ForwardDiff.gradient`
"""
ForwardDiff_gradient(x...) = ForwardDiff.gradient(x...)

"""
`kr(psi, alpha, N)`

Return the van Genuchten relative permeability

# Arguments
- `psi`: pressure head
- `alpha`: van Genucthen's α parameter
- `N`: van Genuchten's N parameter
"""
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

"""
`kr(x)`

Returns `kr(x[1], x[2], x[3])`
"""
kr(x::AbstractArray) = kr(x[1], x[2], x[3])

function NonlinearEquations.Calculus.differentiate(x::NonlinearEquations.Calculus.SymbolParameter{:kr}, args, wrt)
	chain_part = map(x->NonlinearEquations.Calculus.simplify(NonlinearEquations.Calculus.differentiate(x, wrt)), args)
	if chain_part == [0, 0, 0]
		return :(0)
	else
		return :(sum(ForwardDiff_gradient(kr, StaticArrays.SA[$(args...)]) .* StaticArrays.SA[$(chain_part...)]))
	end
end

"""
`hm(x, y)`

Returns the harmonic mean of `x` and `y`
"""
function hm(x, y)
	return 2 / (1 / x + 1 / y)
end

"""
`hm(x)`

Returns the harmonic mean of `x[1]` and `x[2]`
"""
hm(x::AbstractArray) = hm(x[1], x[2])

function NonlinearEquations.Calculus.differentiate(x::NonlinearEquations.Calculus.SymbolParameter{:hm}, args, wrt)
	chain_part = map(x->NonlinearEquations.Calculus.simplify(NonlinearEquations.Calculus.differentiate(x, wrt)), args)
	if chain_part == [0, 0]
		return :(0)
	else
		return :(sum(ForwardDiff_gradient(hm, StaticArrays.SA[$(args...)]) .* StaticArrays.SA[$(chain_part...)]))
	end
end

function NonlinearEquations.Calculus.differentiate(x::NonlinearEquations.Calculus.SymbolParameter{:abs}, args, wrt)
	if length(args) > 1
		error("Too many arguments passed to abs()")
	end
	arg = args[1]
	return :(ifelse($arg > 0, 1, -1) * $(NonlinearEquations.Calculus.differentiate(arg, wrt)))
end

"""
`effective_saturation(alpha::Number, psi::Number, N::Number)`

Return the effective saturation

# Arguments
- `alpha`: van Genucthen's α parameter
- `psi`: pressure head
- `N`: van Genuchten's N parameter
"""
function effective_saturation(alpha::Number, psi::Number, N::Number)
	m = (N - 1) / N
	if psi < 0
		return (1 + abs(alpha * psi) ^ N) ^ (-m)
	else
		return 1
	end
end

@NonlinearEquations.equations exclude=(coords, dirichletnodes, neighbors, areasoverlengths) function richards(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	NonlinearEquations.setnumequations(sum(isfreenode))
	for i = 1:length(Qs)
		if isfreenode[i]
			j = nodei2freenodei[i]
			NonlinearEquations.addterm(j, Qs[i] / (specificstorage[i] * volumes[i]))
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			j1 = nodei2freenodei[node1]
			if isfreenode[node1]  && isfreenode[node2]
				j2 = nodei2freenodei[node2]
				NonlinearEquations.addterm(j1, hm(kr(psi[j1], alphas[i], Ns[i]), kr(psi[j2], alphas[i], Ns[i])) * Ks[i] * (psi[j2] + coords[3, node2] - psi[j1] - coords[3, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			elseif isfreenode[node1] && !isfreenode[node2]
				NonlinearEquations.addterm(j1, hm(kr(psi[j1], alphas[i], Ns[i]), kr(dirichletpsis[node2], alphas[i], Ns[i])) * Ks[i] * (dirichletpsis[node2] + coords[3, node2] - psi[j1] - coords[3, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			end
		end
	end
end
@doc """
`richards_steadystate(Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; callback=soln->nothing, kwargs...)`

Return the residuals at the free nodes of a finite volume discretization of Richards equation

# Arguments
- `psi`: pressure head
- `Ks`: permeability
- `neighbors`: array of pairs indicating which cells share an interface
- `areasoverlengths`: array with the same length as `neighbors` that gives the interfacial area divided by the length between the two cell centers
- `dirichletnodes`: array of indices indicating which nodes have Dirichlet boundary conditions
- `dirichletpsis`: array of pressures at the Dirichlet boundary (length is equal to the number of cells on the grid)
- `coords`: the coordinates of the cell centers
- `alphas`: van Genuchten α parameters for each cell
- `Ns`: van Genuchten N parameters for each cell
- `Qs`: array of fluxes (length is equal to the number of cells on the grid)
- `specificstorage`: array of the specific storage associated with each cell (length is equal to the number of cells on the grid)
- `volumes`: array of the volume of each each cell (length is equal to the number of cells on the grid)
"""
richards_residuals

"""
`richards_steadystate(Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; callback=soln->nothing, kwargs...)`

Return the solution to a steady state unsaturated groundwater problem

# Arguments
- `psi0`: initial guess for the pressure
- `Ks`: permeability
- `neighbors`: array of pairs indicating which cells share an interface
- `areasoverlengths`: array with the same length as `neighbors` that gives the interfacial area divided by the length between the two cell centers
- `dirichletnodes`: array of indices indicating which nodes have Dirichlet boundary conditions
- `dirichletpsis`: array of pressures at the Dirichlet boundary (length is equal to the number of cells on the grid)
- `coords`: the coordinates of the cell centers
- `alphas`: van Genuchten α parameters for each cell
- `Ns`: van Genuchten N parameters for each cell
- `Qs`: array of fluxes (length is equal to the number of cells on the grid)
"""
function richards_steadystate(psi0, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; callback=soln->nothing, kwargs...)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	args = (psi0[isfreenode], Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, ones(length(Qs)), ones(length(Qs)))
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
	df = NLsolve.OnceDifferentiable(residuals!, jacobian!, psi0[isfreenode], residuals0, J0)
	soln = NLsolve.nlsolve(df, psi0[isfreenode]; kwargs...)
	callback(soln)
	if !NLsolve.converged(soln)
		#display(soln)
		error("solution did not converge")
	end
	psifree = soln.zero
	psi = map(i->isfreenode[i] ? psifree[nodei2freenodei[i]] : dirichletpsis[i], 1:length(Qs))
	return psi
end

function ChainRulesCore.rrule(::typeof(richards_steadystate), psi0, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; kwargs...)
	isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
	psi = richards_steadystate(psi0, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs; kwargs...)
	function pullback(delta)
		args = (psi[isfreenode], Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, ones(length(Qs)), ones(length(Qs)))
		lambda = richards_psi(args...)' \ delta[isfreenode]
		req_Ks = richards_Ks(args...)
		req_dirichletpsis = richards_dirichletpsis(args...)
		req_alphas = richards_alphas(args...)
		req_Ns = richards_Ns(args...)
		req_Qs = richards_Qs(args...)
		return (ChainRulesCore.NoTangent(),#step function
				@ChainRulesCore.thunk(zeros(length(psi0))),#psi0 -- should be insensitive to psi0
				@ChainRulesCore.thunk(-(req_Ks' * lambda)),#Ks
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#neighbors
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#areasoverlengths
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#dirichletnodes
				@ChainRulesCore.thunk(-(req_dirichletpsis' * lambda) .+ delta .* (map(x->!x, isfreenode))),#dirichletpsis
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#coords
				@ChainRulesCore.thunk(-(req_alphas' * lambda)),#alphas
				@ChainRulesCore.thunk(-(req_Ns' * lambda)),#Ns
				@ChainRulesCore.thunk(-(req_Qs' * lambda)))#Qs
	end
	return psi, pullback
end
