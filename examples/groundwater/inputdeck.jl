import DPFEHM
import GaussianRandomFields
import PyPlot
import Random
import Zygote

Random.seed!(0)

mins = [0, 0]#meters
maxs = [100, 100]#meters
ns = [101, 101]#number of grid points
coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
Qs = zeros(size(coords, 2))
boundaryhead(x, y) = 5 * (x - maxs[1]) / (mins[1] - maxs[1])
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
for i = 1:size(coords, 2)
	if coords[1, i] == mins[1] || coords[1, i] == maxs[1] || coords[2, i] == mins[2] || coords[2, i] == maxs[2]
		push!(dirichletnodes, i)
		dirichleths[i] = boundaryhead(coords[1:2, i]...)
	end
end

lambda = 50.0#meters -- correlation length of log-conductivity
sigma = 1.0#standard deviation of log-conductivity
mu = 0.0#mean of log-permeability can be arbitrary because this is steady-state and there are no fluid sources

num_eigenvectors = 200
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
@time grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
@time logKs = GaussianRandomFields.sample(grf)
parameterization = hcat(fill(mu, prod(ns)), grf.data.eigenfunc)
parametermeans = [[1.0], zeros(num_eigenvectors)]

#plot a realization
fig, ax = PyPlot.subplots()
ax.imshow(logKs)
display(fig)
println()
PyPlot.close(fig)

#plot the first few eigenvectors and some random ones
fig, axs = PyPlot.subplots(2, 4, figsize=(16, 8))
rp = Random.randperm(size(parameterization, 2))
for (i, ax) in enumerate(axs)
	if i <= length(axs) / 2
		ax.imshow(reshape(parameterization[:, i], ns...))
	else
		ax.imshow(reshape(parameterization[:, rp[i]], ns...))
	end
end
display(fig)
println()
PyPlot.close(fig)

#logKs2Ks_neighbors(Ks) = map(neighbor->exp(0.5 * (Ks[neighbor[1]] + Ks[neighbor[2]])), neighbors)
#logKs2Ks_neighbors(Ks) = exp.([0.5 * (Ks[i1] + Ks[i2]) for (i1, i2) in neighbors])
logKs2Ks_neighbors(Ks) = exp.(Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)])#Zygote differentiates this efficiently but the definitions above are ineffecient with Zygote
function solveforh(logKs, dirichleths)
	@assert length(logKs) == length(Qs)
	Ks_neighbors = logKs2Ks_neighbors(logKs)
	return DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
end

obsnode = findmin(map(i->sum((coords[:, i] .- 0.5 * (mins + maxs)) .^ 2), 1:size(coords, 2)))[2]
f(logKs) = solveforh(logKs, dirichleths)[obsnode]
@show coords[:, obsnode]
h = solveforh(logKs, dirichleths)
h = solveforh(logKs, dirichleths)
print("forward solve time")
@time h = solveforh(logKs, dirichleths)
zg = Zygote.gradient(f, logKs)[1]
zg = Zygote.gradient(f, logKs)[1]
print("gradient time")
@time zg = Zygote.gradient(f, logKs)[1]

#plot the solution, the difference between the solution and the solution for a uniform medium, and the logKs
fig, axs = PyPlot.subplots(1, 4, figsize=(32, 8))
ims = axs[1].imshow(reshape(h, ns...))
fig.colorbar(ims, ax=axs[1])
ims = axs[2].imshow(reshape(h .- map(i->boundaryhead(coords[:, i]...), 1:size(coords, 2)), ns...))
fig.colorbar(ims, ax=axs[2])
ims = axs[3].imshow(logKs)
fig.colorbar(ims, ax=axs[3])
ims = axs[4].imshow(zg)
fig.colorbar(ims, ax=axs[4])
fig.tight_layout()
display(fig)
println()
PyPlot.close(fig)
