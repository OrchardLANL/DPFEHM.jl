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

#=
#test Theis solution against richards equation model
coords_richards = vcat(coords, zeros(size(coords, 2))')
alphas = fill(0.5, length(neighbors))
Ns = fill(1.25, length(neighbors))
function f_richards(u, p, t)
	Ks = unpack(p)
	return DPFEHM.richards_residuals(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, coords_richards, alphas, Ns, Qs, specificstorage, volumes)
end
function f_richards_u(u, p, t)
	Ks = unpack(p)
	return DPFEHM.richards_psi(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, coords_richards, alphas, Ns, Qs, specificstorage, volumes)
end
function f_richards_p(u, p, t)
	Ks = unpack(p)
	return DPFEHM.richards_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, coords_richards, alphas, Ns, Qs, specificstorage, volumes)
end
f_richards_t(u, p, t) = zeros(length(u))
print("richards forward")
@time h_richards = DifferentiableBackwardEuler.steps_diffeq(h0, f_richards, f_richards_u, f_richards_p, f_richards_t, p, t0, tfinal; abstol=1e-6, reltol=1e-6)
richards_drawdowns = -h_richards[end][goodnodes] .+ steadyhead
if doplot
	fig, ax = PyPlot.subplots()
	ax.plot(rs, theis_drawdowns, "r.", ms=20, label="Theis")
	ax.plot(rs, richards_drawdowns, "k", linewidth=3, label="DPFEHM richards")
	ax.set_xlabel("x [m]")
	ax.set_ylabel("drawdown [m]")
	ax.legend()
	display(fig)
	println()
	PyPlot.close(fig)
end
@test isapprox(theis_drawdowns, richards_drawdowns; atol=1e-1)
@test isapprox(h_richards[:, :], h_gw[:, :])#make sure richards and groundwater are giving the same thing
g_richards(p) = DifferentiableBackwardEuler.steps(h0, f_richards, f_richards_u, f_richards_p, f_richards_t, p, t0, tfinal; abstol=1e-6, reltol=1e-6)[goodnodes[round(Int, 0.25 * end)], end]
print("richards gradient")
@time grad_richards_zygote = Zygote.gradient(g_richards, p)[1]
checkgradientquickly(g_richards, p, grad_richards_zygote, 3; delta=1e-8, rtol=1e-1)
=#

