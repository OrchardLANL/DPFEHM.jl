function getpoints(min, max, n)
	return range(min, max; length=n)
end
function regulargrid2d(mins, maxs, ns, dz)
	linearindex = (i1, i2)->i2 + ns[2] * (i1 - 1)
	coords = Array{Float64}(undef, 2, prod(ns))
	xs = getpoints(mins[1], maxs[1], ns[1])
	ys = getpoints(mins[2], maxs[2], ns[2])
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

function regulargrid3d(mins, maxs, ns)
	linearindex = (i1, i2, i3)->i3 + ns[3] * (i2 - 1) + ns[3] * ns[2] * (i1 - 1)
	coords = Array{Float64}(undef, 3, prod(ns))
	xs = getpoints(mins[1], maxs[1], ns[1])
	ys = getpoints(mins[2], maxs[2], ns[2])
	zs = getpoints(mins[3], maxs[3], ns[3])
	dx = xs[2] - xs[1]
	dy = ys[2] - ys[1]
	dz = zs[2] - zs[1]
	j = 1
	neighbors = Array{Pair{Int, Int}}(undef, 3 * prod(ns) - ns[1] * ns[2] - ns[1] * ns[3] - ns[2] * ns[3])
	areasoverlengths = Array{Float64}(undef, 3 * prod(ns) - ns[1] * ns[2] - ns[1] * ns[3] - ns[2] * ns[3])
	volumes = fill(dx * dy * dz, size(coords, 2))
	for i1 = 1:ns[1]
		for i2 = 1:ns[2]
			for i3 = 1:ns[3]
				coords[1, linearindex(i1, i2, i3)] = xs[i1]
				coords[2, linearindex(i1, i2, i3)] = ys[i2]
				coords[3, linearindex(i1, i2, i3)] = zs[i3]
				if i1 < ns[1]
					neighbors[j] = linearindex(i1, i2, i3)=>linearindex(i1 + 1, i2, i3)
					areasoverlengths[j] = dy * dz / dx
					j += 1
				end
				if i2 < ns[2]
					neighbors[j] = linearindex(i1, i2, i3)=>linearindex(i1, i2 + 1, i3)
					areasoverlengths[j] = dx * dz / dy
					j += 1
				end
				if i3 < ns[3]
					neighbors[j] = linearindex(i1, i2, i3)=>linearindex(i1, i2, i3 + 1)
					areasoverlengths[j] = dx * dy / dz
					j += 1
				end
			end
		end
	end
	return coords, neighbors, areasoverlengths, volumes
end

function darcy_velocity(h, Ks, mins, maxs, ns)#note: this is currently not very Zygote-compatible
	allpoints = map(getpoints, mins, maxs, ns)
	h_itp_unscaled = Interpolations.interpolate(h, Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line(Interpolations.OnCell()))))
	h_itp = Interpolations.scale(h_itp_unscaled, reverse(allpoints)...)
	K_itp_unscaled = Interpolations.interpolate(Ks, Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line(Interpolations.OnCell()))))
	K_itp = Interpolations.scale(K_itp_unscaled, reverse(allpoints)...)
	return (x...)->reverse(K_itp(x...) * Interpolations.gradient(h_itp, x...))
end
