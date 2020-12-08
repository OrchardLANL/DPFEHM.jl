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

function regulargrid3d(mins, maxs, ns)
	linearindex = (i1, i2, i3)->i3 + ns[3] * (i2 - 1) + ns[3] * ns[2] * (i1 - 1)
	coords = Array{Float64}(undef, 3, prod(ns))
	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	zs = range(mins[3]; stop=maxs[3], length=ns[3])
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

function darcy_velocity(h, Ks, coords, ns)
	@assert length(ns) == 2
	@show size(Ks)
	@show size(ns)
	hr = reshape(h, ns...)
	vr = zeros(2, ns...)
	@show size(vr)
	@show ns
	coordsr = reshape(coords, 2, reverse(ns)...)
	@show size(coordsr)
	@show size(hr)
	for i = 2:ns[1] - 1
		for j = 2:ns[2] - 1
			#=
			if coordsr[1, j, i + 1, j] - coordsr[1, i - 1, j] == 0
				@show i, j
				@show coordsr[1, i + 1, j], coordsr[1, i - 1, j]
			end
			=#
			vr[1, i, j] = Ks[i, j] * (hr[i + 1, j] - hr[i - 1, j]) / (coordsr[1, j, i + 1] - coordsr[1, j, i - 1])
			vr[2, i, j] = Ks[i, j] * (hr[i, j + 1] - hr[i, j - 1]) / (coordsr[2, j + 1, i] - coordsr[2, j - 1, i])
		end
	end
	return vr
end
