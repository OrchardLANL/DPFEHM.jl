function regulargrid2d(mins, maxs, ns, dz)
	linearindex = (i1, i2)->i2 + ns[2] * (i1 - 1)
	coords = Array{Float64}(undef, 2, prod(ns))
	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	dx = xs[2] - xs[1]
	dy = ys[2] - ys[1]
	j = 1
	neighbors = Array{Pair{Int, Int}}(undef, 2 * prod(ns) - ns[1] - ns[2])
	areasoverlengths = Array{Float64}(undef, 2 * prod(ns) - ns[1] - ns[2])
	volumes = Array{Float64}(undef, 0)
	for i1 = 1:ns[1]
		for i2 = 1:ns[2]
			push!(volumes, dx * dy * dz)
			coords[1, linearindex(i1, i2)] = xs[i1]
			coords[2, linearindex(i1, i2)] = ys[i2]
			if i1 < ns[1]
				neighbors[j] = linearindex(i1, i2)=>linearindex(i1 + 1, i2)
				areasoverlengths[j] = dy * dz / dx
				j += 1
			end
			if i2 < ns[2]
				neighbors[j] = linearindex(i1, i2)=>linearindex(i1, i2 + 1)
				areasoverlengths[j] = dx * dz / dy
				j += 1
			end
		end
	end
	return coords, neighbors, areasoverlengths, volumes
end

