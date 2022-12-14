"""
`getfreenodes(n, dirichletnodes)`

Returns an array indicating whether or not a node is free, a dictionary that maps a global index into a free node index, and an array that maps a free node index into a global index. A free node is one that is not defined by a Dirichlet boundary coundition. That is, a free node, is one that is not listed in `dirichletnodes`.

# Arguments
- `n`: the number of cells on the mesh
- `dirichletnodes`: the indices of the cells that are constrained by a Dirichlet boundary condition
"""
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

"""
`addboundaryconditions(hfree, dirichletnodes, dirichleths, isfreenode, nodei2freenodei)`

Returns an array giving the pressures at all nodes, including the nodes with Dirichlet boundary conditions.

# Arguments
- `hfree`: an array giving values of h at each cell that is not constrained by a Dirichlet boundary condition
- `dirichletnodes`: the indices of the cells that are constrained by a Dirichlet boundary condition
- `dirichleth`: the values of h at the cells with Dirichlet boundary conditions
- `isfreenode`: a boolean array such that isfreenode[i] is true if cell i does not have a Dirichlet boundary condition
- `nodei2freenodei`: a dictionary that maps a node index to a free node index
"""
function addboundaryconditions(hfree, dirichletnodes, dirichleths, isfreenode, nodei2freenodei)
	return map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(dirichleths))
end
