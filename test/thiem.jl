using Test
import DPFEHM
import Zygote

doplot = false
if doplot == true
	import PyPlot
end

function thiem(R, r, T, Q)
	return Q * log(R / r) / (2 * pi * T)
end
#utitlity for checking gradients
function checkgradientquickly(f, x0, gradf, n; delta::Float64=1e-8, kwargs...)
	indicestocheck = sort(collect(1:length(x0)), by=i->abs(gradf[i]), rev=true)[1:n]
	#indicestocheck = [indicestocheck; rand(1:length(x0), n)]
	f0 = f(x0)
	for i in indicestocheck
		x = copy(x0)
		x[i] += delta
		fval = f(x)
		grad_f_i = (fval - f0) / delta
		@test isapprox(gradf[i], grad_f_i; kwargs...)
	end
end

#test Thiem solution against groundwater model
steadyhead = 1e3
sidelength = 50.0
thickness = 10.0
R = sidelength
n = 101
ns = [n, n]
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d([-sidelength, -sidelength], [sidelength, sidelength], ns, thickness)
k = 1e-5
Ks = fill(k, length(neighbors))
Q = 1e-3
Qs = zeros(size(coords, 2))
Qs[ns[2] * (div(ns[1] + 1, 2) - 1) + div(ns[2] + 1, 2)] = -Q#put a fluid source in the middle
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
for i = 1:size(coords, 2)
	if sqrt(coords[1, i] ^ 2 + coords[2, i] ^ 2) >= R
		push!(dirichletnodes, i)
		dirichleths[i] = steadyhead
	end
end
print("steady state groundwater forward")
@time h_gw = DPFEHM.groundwater_steadystate(Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; reltol=1e-12)
r0 = 0.1
goodnodes = collect(filter(i->coords[2, i] == 0 && coords[1, i] > r0, 1:size(coords, 2)))
rs = coords[1, goodnodes]
T = thickness * k
thiem_drawdowns = thiem.(R, rs, T, Q)
gw_drawdowns = -h_gw[goodnodes] .+ steadyhead
if doplot
	fig, ax = PyPlot.subplots()
	ax.plot(rs, thiem_drawdowns, "r.", ms=20, label="Thiem")
	ax.plot(rs, gw_drawdowns, "k", linewidth=3, label="DPFEHM groundwater")
	ax.set_xlabel("x [m]")
	ax.set_ylabel("drawdown [m]")
	ax.legend()
	display(fig)
	println()
	PyPlot.close(fig)
end
@test isapprox(thiem_drawdowns, gw_drawdowns; rtol=1e-1)
g_gw(Ks) = DPFEHM.groundwater_steadystate(Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; reltol=1e-12)[goodnodes[round(Int, 0.25 * end)], end]
print("groundwater gradient")
@time grad_gw_zygote = Zygote.gradient(g_gw, Ks)[1]
checkgradientquickly(g_gw, Ks, grad_gw_zygote, 3; delta=1e-8, rtol=1e-1)
