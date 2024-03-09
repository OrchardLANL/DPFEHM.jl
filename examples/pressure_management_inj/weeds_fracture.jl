import DPFEHM
import GaussianRandomFields
import PyPlot
import Zygote

plotstuff = false

module FractureMaker
mutable struct FractureTip
    i::Int
    j::Int
    di::Int
    dj::Int
    alive::Bool
end
struct Material
    isfractured::Matrix{Int}
end
function Material(n, m)
    return Material(zeros(n, m))
end
inside(m::Material, tip::FractureTip) = inside(m.isfractured, tip)
function inside(A::Matrix, tip::FractureTip)
    return tip.i <= size(A, 1) && tip.i > 0 && tip.j <= size(A, 2) && tip.j > 0
end
function timestep!(m::Material, tips::Array{FractureTip})
    for tip in tips
        if inside(m, tip) && tip.alive
            m.isfractured[tip.i, tip.j] = true
        end
        if tip.alive
            tip.i += tip.di
            tip.j += tip.dj
        end
        if inside(m, tip) && m.isfractured[tip.i, tip.j] == true
            tip.alive = false
        end
    end
end
function simulate(mat::Material, tips, numtimesteps)
    for k = 1:numtimesteps
        timestep!(mat, tips)
    end
    return mat, tips
end
end # module FractureMaker

# Set up the mesh
ns = [101, 201]#mesh size -- make it at least 100x200 to have the resolution for the fractures
steadyhead = 0e0 #meters
width = 1000 #meters
heights = [200, 100, 200]
thickness  = 1.0 #meters
mins = [0, 0] #meters
maxs = [width, sum(heights)] #meters
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
# Set up the eigenvector parameterization of the geostatistical log-permeability field
num_eigenvectors = 200
sigma = 1.0
lambda = 50
mean_log_conductivities = [log(1e-4), log(1e-8), log(1e-3)] #log(1e-4 [m/s]) -- similar to a pretty porous oil reservoir
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
parameterization = copy(grf.data.eigenfunc)
sigmas = copy(grf.data.eigenval)
masks = [zeros(Bool, prod(ns)) for i = 1:3]
for i = 1:size(coords, 2)
    if coords[2, i] > heights[1] + heights[2]
        masks[3][i] = true
    elseif coords[2, i] > heights[1]
        masks[2][i] = true
    else
        masks[1][i] = true
    end
end
masks = map(x->reshape(x, reverse(ns)...), masks)
if plotstuff
    fig, axs = PyPlot.subplots(1, 3)
    for i = 1:3
        axs[i].imshow(masks[i], origin="lower")
    end
    display(fig); println()
    PyPlot.close(fig)
end
#setup the fracture network
material = FractureMaker.Material(reverse(ns)...)
num_horizontal_fractures = sum(heights) ÷ 30#put a horizontal fracture every 30 meters or so
tips = Array{FractureMaker.FractureTip}(undef, 2 * num_horizontal_fractures)
for i = 1:num_horizontal_fractures
    j = rand(1:ns[1])
    tips[2 * i - 1] = FractureMaker.FractureTip(5 + (i - 1) * 30 * ns[2] ÷ sum(heights), j, 0, 1, true)
    tips[2 * i] = FractureMaker.FractureTip(5 + (i - 1) * 30 * ns[2] ÷ sum(heights), j, 0, -1, true)
end
FractureMaker.simulate(material, tips, 9 * ns[1] ÷ 20)#the time steps let the fracture grow to be at most a little less than the width of the domain
horizontal_fracture_mask = copy(material.isfractured) .* masks[2]
num_vertical_fractures = width ÷ 10#put a vertical fracture every 10 meters or so
tips = Array{FractureMaker.FractureTip}(undef, 2 * num_vertical_fractures)
for i = 1:num_vertical_fractures
    j = rand(1:ns[1])
    k = rand(1:ns[2])
    tips[2 * i - 1] = FractureMaker.FractureTip(k, j, 1, 0, true)
    tips[2 * i] = FractureMaker.FractureTip(k, j, -1, 0, true)
end
FractureMaker.simulate(material, tips, ns[2] * heights[2] ÷ (3 * sum(heights)))
vertical_fracture_mask = copy(material.isfractured) .* masks[2] - horizontal_fracture_mask
if plotstuff
    fig, axs = PyPlot.subplots(1, 2)
    axs[1].imshow(horizontal_fracture_mask, origin="lower")
    axs[2].imshow(vertical_fracture_mask, origin="lower")
    display(fig); println()
    PyPlot.close(fig)
end
#play with these four numbers to make the fractures more or less likely to enable lots of flow from the lower reservoir to the upper reservoir
horizontal_fracture_mean_conductivity = log(1e-10)
vertical_fracture_mean_conductivity = log(1e-10)
horizontal_fracture_sigma_conductivity = 10
vertical_fracture_sigma_conductivity = 10
x2logKs(x) = log.(exp.(sum(masks[i]' .* reshape(parameterization * (sigmas .* x[(i - 1) * num_eigenvectors + 1:i * num_eigenvectors]) .+ mean_log_conductivities[i], ns...) for i = 1:3)') + horizontal_fracture_mask * exp(horizontal_fracture_mean_conductivity + horizontal_fracture_sigma_conductivity * x[end - 1]) + vertical_fracture_mask * exp(vertical_fracture_mean_conductivity + vertical_fracture_sigma_conductivity * x[end]))
logKs = x2logKs(randn(3 * num_eigenvectors + 2))
if plotstuff
    fig, ax = PyPlot.subplots()
    ax.imshow(logKs, origin="lower")
    display(fig); println()
    PyPlot.close(fig)
end

# Make the boundary conditions be dirichlet with a fixed head on the left and right boundaries with zero flux on the top and bottom
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
for i = 1:size(coords, 2)
    if coords[1, i] == width || coords[1, i] == 0
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
injection_node, critical_point_node = get_nodes_near(coords, [[width / 2, heights[1] / 2], [width / 2, heights[1] + heights[2] + 0.5 * heights[3]]]) #put the critical point near (-80, -80) and the injection node near (80, 80)
injection_nodes = [injection_node]
# Set the Qs for the whole field to zero, then set it to the injection rate divided by the number of injection nodes at the injection nodes
Qs = zeros(size(coords, 2))
Qinj = 0.031688 #injection rate [m^3/s] (1 MMT water/yr)
Qs[injection_nodes] .= Qinj / length(injection_nodes)
# Set up the function that solves the flow equations and outputs the relevant pressure
logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#Zygote differentiates this efficiently but the definitions above are ineffecient with Zygote
function plotit(result, logKs)
    if plotstuff
        fig, axs = PyPlot.subplots(1, 2)
        axs[1].imshow(reshape(result, reverse(ns)...), origin="lower")
        axs[2].imshow(logKs, origin="lower")
        display(fig); println()
        PyPlot.close(fig)
    end
end
Zygote.@nograd plotit
function solveforh(logKs)
    @assert length(logKs) == length(Qs)
    Ks_neighbors = logKs2Ks_neighbors(logKs)
    result = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; linear_solver=DPFEHM.cholesky_solver)
    plotit(result, logKs)
    return result
end
solveforheigs(x) = solveforh(x2logKs(x))
# Set up the functions that compute the pressure at the critical point and the gradient of the pressure at the critical point with respect to the eigenvector coefficients
f(x) = solveforh(x2logKs(x))[critical_point_node]*9.807*997*1e-6 #convert from head (meters) to pressure (MPa) (for water 25 C)
grad_f(x) = Zygote.gradient(f, x)[1]
