#solve u_t + div(v * u) - div(D * grad(u)) - Q = 0 on an unstructured mesh using an upwind discretization of the velocity term (or downwind if use_upwind=false)
@NonlinearEquations.equations exclude=(areasoverlengths, volumes, dirichletnodes, coords) function transport(u, vxs, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords; use_upwind=true)
	@assert size(coords, 1) == 3#it is assumed that this is a 3d problem
	NonlinearEquations.setnumequations(length(u))
	for i = 1:length(u)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, u[i] - dirichletus[i])
		else
			NonlinearEquations.addterm(i, Qs[i] / (volumes[i]))
		end
	end
	delta = zeros(eltype(coords), size(coords, 1))
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			#add the advection term
			delta .= coords[:, node1] .- coords[:, node2]
			is_upwind = ((vxs[i] * delta[1] + vys[i] * delta[2] + vzs[i] * delta[3]) > 0)
			if (is_upwind && use_upwind) || (!is_upwind && !use_upwind)#!use_upwind basically means use_downwind and !is_upwind basically means is_downwind
				if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
					NonlinearEquations.addterm(node1, -(vxs[i] * delta[1] + vys[i] * delta[2] + vzs[i] * delta[3]) * (u[node1] - u[node2]) * areasoverlengths[i] / (volumes[node1]))
				elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
					NonlinearEquations.addterm(node1, -(vxs[i] * delta[1] + vys[i] * delta[2] + vzs[i] * delta[3]) * (u[node1] - dirichletus[node2]) * areasoverlengths[i] / (volumes[node1]))
				end
			end
			#add the diffusion term
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, Ds[i] * (u[node2] - u[node1]) * areasoverlengths[i] / (volumes[node1]))
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, Ds[i] * (dirichletus[node2] - u[node1]) * areasoverlengths[i] / (volumes[node1]))
			end
		end
	end
end
