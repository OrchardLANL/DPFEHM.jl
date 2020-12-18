import ChainRulesCore
import DifferentiableBackwardEuler
import DPFEHM
import GaussianRandomFields
import Optim
import PyPlot
import Random
import Zygote

Random.seed!(0)

#set up the grid
mins = [0, 0, 0]; maxs = [50, 50, 5]#size of the domain, in meters
ns = [100, 100, 10]#number of nodes on the grid
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid3d(mins, maxs, ns)#build the grid

#set up the boundary conditions
Qs = zeros(size(coords, 2))
injectionnode = 1#inject in the lower left corner
Qs[injectionnode] = 1e-4#m^3/s
dirichletnodes = Int[size(coords, 2)]#fix the pressure in the upper right corner
dirichleths = zeros(size(coords, 2))
dirichleths[size(coords, 2)] = 0.0

#set up the initial condition
h0 = zeros(size(coords, 2))

#set up the conductivity field
lambda = 50.0#meters -- correlation length of log-conductivity
sigma = 1.0#standard deviation of log-conductivity
mu = -9.0#mean of log conductivity -- ~1e-4 m/s, like clean sand here https://en.wikipedia.org/wiki/Hydraulic_conductivity#/media/File:Groundwater_Freeze_and_Cherry_1979_Table_2-2.png
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
num_eigenvectors = 200
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
logKs = zeros(reverse(ns)...)
logKs2d = mu .+ GaussianRandomFields.sample(grf)'#generate a random realization of the log-conductivity field
for i = 1:ns[3]#copy the 2d field to each of the 3d layers
	v = view(logKs, i, :, :)
	v .= logKs2d
end

#set up the storage
specificstorage = fill(0.1, size(coords, 2))

#plot the log-conductivity
fig, ax = PyPlot.subplots()
img = ax.imshow(logKs[1, :, :], origin="lower")
ax.title.set_text("Conductivity Field")
fig.colorbar(img)
display(fig)
println()
PyPlot.close(fig)

logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
function solveforh(logKs, dirichleths)
	@assert length(logKs) == length(Qs)
	Ks_neighbors = logKs2Ks_neighbors(logKs)
	p = [Ks_neighbors; dirichleths]
	h_gw = DifferentiableBackwardEuler.steps(h0, f_gw, f_gw_u, f_gw_p, f_gw_t, p, 0.0, 1.0 * 60 * 60 * 24 * 365 * 10; abstol=1e-1, reltol=1e-1)
	return h_gw
end
hflat2h3d(h) = reshape(h, reverse(ns)...)

function unpack(p)
	@assert length(p) == length(neighbors) + size(coords, 2)
	Ks = p[1:length(neighbors)]
	dirichleths = p[length(neighbors) + 1:length(neighbors) + size(coords, 2)]
	return Ks, dirichleths
end

#set up some functions needed by diffeq
function f_gw(u, p, t)
	Ks, dirichleths = unpack(p)
	return DPFEHM.groundwater_residuals(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end
function f_gw_u(u, p, t)
	Ks, dirichleths = unpack(p)
	return DPFEHM.groundwater_h(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end
function f_gw_p(u, p, t)
	Ks, dirichleths = unpack(p)
	J1 = DPFEHM.groundwater_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
	J2 = DPFEHM.groundwater_dirichleths(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
	return hcat(J1, J2)
end
f_gw_t(u, p, t) = zeros(length(u))

#now do a forward solve for the head
print("forward solve time")
@time h = solveforh(logKs, dirichleths)#solve for the head
#plot the head at the bottom of the domain
fig, ax = PyPlot.subplots()
img = ax.imshow(hflat2h3d(h[:, end])[1, :, :], origin="lower")
ax.title.set_text("Head")
fig.colorbar(img)
display(fig)
println()
PyPlot.close(fig)

#now compute the gradient of a function involving solveforh
gradient_node = div(size(coords, 2), 2) + 500
gradient_node_x = coords[1, gradient_node]
gradient_node_y = coords[2, gradient_node]
print("forward and gradient time")
@time grad = Zygote.gradient((x, y)->hflat2h3d(solveforh(x, y)[:, end])[gradient_node], logKs, dirichleths)#calculate the gradient (which involves a redundant calculation of the forward pass)
function_evaluation, back = Zygote.pullback((x, y)->hflat2h3d(solveforh(x, y)[:, end])[gradient_node], logKs, dirichleths)#this pullback thing lets us not redo the forward pass
print("gradient time")
@time grad = back(1.0)#compute the gradient of a function involving solveforh
#plot the gradient of the function w.r.t. the logK at the bottom of the domain
fig, ax = PyPlot.subplots()
img = ax.imshow(grad[1][1, :, :], origin="lower", extent=[mins[1], maxs[1], mins[2], maxs[2]])
ax.plot([gradient_node_x], [gradient_node_y], "r.", ms=10, alpha=0.5)
ax.title.set_text("Gradient of head at dot w.r.t. logK at bottom of domain")
fig.colorbar(img)
display(fig)
println()
PyPlot.close(fig)
