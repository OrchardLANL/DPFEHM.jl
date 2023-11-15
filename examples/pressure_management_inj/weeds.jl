import DPFEHM
import GaussianRandomFields
import PyPlot
import Zygote

# Set up the mesh
n = 51 #the mesh will be 51 by 51
ns = [n, n]
steadyhead = 0e0 #meters
sidelength = 200 #meters
thickness  = 1.0 #meters
mins = [-sidelength, -sidelength] #meters
maxs = [sidelength, sidelength] #meters
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
# Set up the eigenvector parameterization of the geostatistical log-permeability field
num_eigenvectors = 200
sigma = 1.0
lambda = 50
mean_log_conductivity = log(1e-4) #log(1e-4 [m/s]) -- similar to a pretty porous oil reservoir
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
logKs = GaussianRandomFields.sample(grf)
parameterization = copy(grf.data.eigenfunc)
sigmas = copy(grf.data.eigenval)
# Make the boundary conditions be dirichlet with a fixed head at all boundaries
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
for i = 1:size(coords, 2)
    if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end
# Set up the location of the injection and critical point
function get_nodes_near(coords, obslocs)
	obsnodes = Array{Int}(undef, length(obslocs))
	for i = 1:length(obslocs)
		obsnodes[i] = findmin(map(j->sum((obslocs[i] .- coords[:, j]) .^ 2), 1:size(coords, 2)))[2]
	end
	return obsnodes
end
critical_point_node, injection_node = get_nodes_near(coords, [[-80, -80], [80, 80]]) #put the critical point near (-80, -80) and the injection node near (80, 80)
injection_nodes = [injection_node]
# Set the Qs for the whole field to zero, then set it to the injection rate divided by the number of injection nodes at the injection nodes
Qs = zeros(size(coords, 2))
Qinj = 0.031688 #injection rate [m^3/s] (1 MMT water/yr)
Qs[injection_nodes] .= Qinj / length(injection_nodes)
# Set up the function that solves the flow equations and outputs the relevant pressure
logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#Zygote differentiates this efficiently but the definitions above are ineffecient with Zygote
function solveforh(logKs)
    @assert length(logKs) == length(Qs)
    if maximum(logKs) - minimum(logKs) > 25
        return fill(NaN, length(Qs)) #this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
    else
        Ks_neighbors = logKs2Ks_neighbors(logKs)
        return DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
    end
end
x2logKs(x) = reshape(parameterization * (sigmas .* x), ns...) .+ mean_log_conductivity
solveforheigs(x) = solveforh(x2logKs(x))
# Set up the functions that compute the pressure at the critical point and the gradient of the pressure at the critical point with respect to the eigenvector coefficients
f(x) = solveforh(x2logKs(x))[critical_point_node]*9.807*997*1e-6 #convert from head (meters) to pressure (MPa) (for water 25 C)
grad_f(x) = Zygote.gradient(f, x)[1]