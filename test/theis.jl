using Test
import DifferentiableBackwardEuler
import DPFEHM
import Zygote

doplot = false
if doplot == true
	import PyPlot
end

#Theis solution
function W(u)
	if u <= 1
		return -log(u) + -0.57721566 + 0.99999193u^1 + -0.24991055u^2 + 0.05519968u^3 + -0.00976004u^4 + 0.00107857u^5
	else
		return (u^2 + 2.334733u + 0.250621) / (u^2 + 3.330657u + 1.681534) * exp(-u) / u
	end
end
function theis(t, r, T, S, Q)
	return Q * W(r^2 * S / (4 * T * t)) / (4 * pi * T)
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

#test Theis solution against groundwater model
steadyhead = 1e3
sidelength = 50.0
thickness = 10.0
n = 101
ns = [n, n]
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d([-sidelength, -sidelength], [sidelength, sidelength], ns, thickness)
k = 1e-5
Ks = fill(k, length(neighbors))
T = thickness * k
Q = 1e-3
Ss = 0.1
specificstorage = fill(Ss, size(coords, 2))
S = Ss * thickness
Qs = zeros(size(coords, 2))
Qs[ns[2] * (div(ns[1] + 1, 2) - 1) + div(ns[2] + 1, 2)] = -Q#put a fluid source in the middle
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
for i = 1:size(coords, 2)
	if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
		push!(dirichletnodes, i)
		dirichleths[i] = steadyhead
	end
end
isfreenode, nodei2freenodei, freenodei2nodei = DPFEHM.getfreenodes(prod(ns), dirichletnodes)
function unpack(p)
	@assert length(p) == length(neighbors)
	Ks = p[1:length(neighbors)]
	return Ks
end
function f_gw(u, p, t)
	Ks = unpack(p)
	return -DPFEHM.groundwater_residuals(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end
function f_gw_u(u, p, t)
	Ks = unpack(p)
	return -DPFEHM.groundwater_h(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end
function f_gw_p(u, p, t)
	Ks = unpack(p)
	return -DPFEHM.groundwater_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end
f_gw_t(u, p, t) = zeros(length(u))
t0 = 0.0
tfinal = 60 * 60 * 24 * 1e1
h0 = fill(steadyhead, size(coords, 2) - length(dirichletnodes))
p = Ks
print("groundwater forward")
@time h_gw = DifferentiableBackwardEuler.steps_diffeq(h0, f_gw, f_gw_u, f_gw_p, f_gw_t, p, t0, tfinal; abstol=1e-6, reltol=1e-6)
dpfehm_solution = map(h->DPFEHM.addboundaryconditions(h, dirichletnodes, dirichleths, isfreenode, nodei2freenodei), h_gw.u)
r0 = 0.1
goodnodes = collect(filter(i->coords[2, i] == 0 && coords[1, i] > r0 && coords[1, i] <= sidelength / 2, 1:size(coords, 2)))
rs = coords[1, goodnodes]
theis_drawdowns = theis.(h_gw.t[end], rs, T, S, Q)
gw_drawdowns = -dpfehm_solution[end][goodnodes] .+ steadyhead
if doplot
	fig, ax = PyPlot.subplots()
	ax.plot(rs, theis_drawdowns, "r.", ms=20, label="Theis")
	ax.plot(rs, gw_drawdowns, "k", linewidth=3, label="DPFEHM groundwater")
	ax.set_xlabel("x [m]")
	ax.set_ylabel("drawdown [m]")
	ax.legend()
	display(fig)
	println()
	PyPlot.close(fig)
end
@test isapprox(theis_drawdowns, gw_drawdowns; atol=1e-1)
g_gw(p) = DifferentiableBackwardEuler.steps(h0, f_gw, f_gw_u, f_gw_p, f_gw_t, p, t0, tfinal; abstol=1e-6, reltol=1e-6)[nodei2freenodei[goodnodes[round(Int, 0.25 * end)]], end]
print("groundwater gradient")
@time grad_gw_zygote = Zygote.gradient(g_gw, p)[1]
checkgradientquickly(g_gw, p, grad_gw_zygote, 3; delta=1e-8, rtol=1e-1)

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
h0 = fill(steadyhead, size(coords, 2) - length(dirichletnodes))
@time h_richards = DifferentiableBackwardEuler.steps_diffeq(h0, f_richards, f_richards_u, f_richards_p, f_richards_t, p, t0, tfinal; abstol=1e-6, reltol=1e-6)
isfreenode, nodei2freenodei, freenodei2nodei = DPFEHM.getfreenodes(length(Qs), dirichletnodes)
richards_dpfehm_solution = map(h->DPFEHM.addboundaryconditions(h, dirichletnodes, dirichleths, isfreenode, nodei2freenodei), h_richards.u)
richards_drawdowns = -richards_dpfehm_solution[end][goodnodes] .+ steadyhead
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
#@test isapprox(h_richards[isfreenode, :], h_gw[:, :])#make sure richards and groundwater are giving the same thing -- commenting this out for now because the recent changes in gw mean they shouldn't be exactly the same -- uncomment when those changes get propagated into richards as well
g_richards(p) = DifferentiableBackwardEuler.steps(h0, f_richards, f_richards_u, f_richards_p, f_richards_t, p, t0, tfinal; abstol=1e-6, reltol=1e-6)[goodnodes[round(Int, 0.25 * end)], end]
print("richards gradient")
@time grad_richards_zygote = Zygote.gradient(g_richards, p)[1]
checkgradientquickly(g_richards, p, grad_richards_zygote, 3; delta=1e-8, rtol=1e-1)
