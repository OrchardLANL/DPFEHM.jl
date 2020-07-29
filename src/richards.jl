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

function saturation(psi::Array, coords::Array)
	Se = similar(psi)
	for i = 1:length(psi)
		if inclay(coords[1, i], coords[2, i])
			_, alpha, N, residual_saturation = params[:clay]
		else
			_, alpha, N, residual_saturation = params[:claysilt]
		end
		Se[i] = residual_saturation + effective_saturation(alpha, psi[i], N) * (1 - residual_saturation)
	end
	return Se
end

@NonlinearEquations.equations exclude=(coords, dirichletnodes, neighbors, areasoverlengths) function richards(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
	NonlinearEquations.setnumequations(length(psi))
	for i = 1:length(psi)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, psi[i] - dirichletpsis[i])
		else
			NonlinearEquations.addterm(i, Qs[i] / (specificstorage[i] * volumes[i]))
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, hm(kr(psi[node1], alphas[i], Ns[i]), kr(psi[node2], alphas[i], Ns[i])) * Ks[i] * (psi[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, hm(kr(psi[node1], alphas[i], Ns[i]), kr(dirichletpsis[node2], alphas[i], Ns[i])) * Ks[i] * (dirichletpsis[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			end
		end
	end
end
