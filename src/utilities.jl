function getfreenodes(n, dirichletnodes)
	isfreenode = fill(true, n)
	isfreenode[dirichletnodes] .= false
	nodei2freenodei = fill(-1, length(isfreenode))
	freenodei2nodei = Array{Int}(undef, sum(isfreenode))
	j = 1
	for i = 1:length(isfreenode)
		if isfreenode[i]
			nodei2freenodei[i] = j
			freenodei2nodei[j] = i
			j += 1
		end
	end
	return isfreenode, nodei2freenodei, freenodei2nodei
end

function addboundaryconditions(hfree, dirichletnodes, dirichleths, isfreenode, nodei2freenodei)
	return map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(dirichleths))
end
