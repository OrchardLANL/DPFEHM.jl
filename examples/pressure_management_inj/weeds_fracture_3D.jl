import DPFEHM
import GaussianRandomFields
import PyPlot
import Zygote

plotstuff = true

module FractureMaker
mutable struct FractureTip
    i::Int
    j::Int
    k::Int
    di::Int
    dj::Int
    dk::Int
    alive::Bool
end

struct Material
    isfractured::Array{Float64, 3}
end

function Material(n, m,p)
    return Material(zeros(n, m,p))
end

inside(m::Material, tip::FractureTip) = inside(m.isfractured, tip)

function inside(A::Array{Float64, 3}, tip::FractureTip)
    if tip.di ==0
        return tip.i <= size(A, 1) && tip.i > 0 && tip.j <= size(A, 2) && tip.j > 0 && tip.k <= size(A, 3) && tip.k > 0 && (tip.k +1) <= size(A, 2) && (tip.k + 1)> 0 && (tip.k -1) <= size(A, 2) && (tip.k - 1)> 0 && (tip.k +2) <= size(A, 2) && (tip.k +2)> 0 && (tip.k -2) <= size(A, 2) && (tip.k -2)> 0
    else
        return tip.i <= size(A, 1) && tip.i > 0 && tip.j <= size(A, 2) && tip.j > 0 && tip.k <= size(A, 3) && tip.k > 0 && (tip.k +1) <= size(A, 2) && (tip.k + 1)> 0 && (tip.k -1) <= size(A, 2) && (tip.k - 1)> 0 && (tip.k +2) <= size(A, 2) && (tip.k +2)> 0 && (tip.k -2) <= size(A, 2) && (tip.k -2)> 0
    end
end
    
function timestep!(m::Material, tips::Array{FractureTip})
    for tip in tips
        if inside(m, tip) && tip.alive
            if tip.di == 0
                m.isfractured[tip.i, tip.j, :] .= true
                #=
                m.isfractured[tip.i, tip.j, tip.k] = true
                m.isfractured[tip.i, tip.j, tip.k+1] = true
                m.isfractured[tip.i, tip.j, tip.k-1] = true
                m.isfractured[tip.i, tip.j, tip.k+2] = true
                m.isfractured[tip.i, tip.j, tip.k-2] = true
                =#
            else
                m.isfractured[tip.i, tip.j, :] .= true
                #=
                m.isfractured[tip.i, tip.j, tip.k] = true
                m.isfractured[tip.i, tip.j, tip.k+1] = true
                m.isfractured[tip.i, tip.j, tip.k-1] = true
                m.isfractured[tip.i, tip.j, tip.k+2] = true
                m.isfractured[tip.i, tip.j, tip.k-2] = true
                =#
            end
        end
        if tip.alive
            tip.i += tip.di
            tip.j += tip.dj
            tip.k += tip.dk
        end
        if inside(m, tip) && m.isfractured[tip.i, tip.j, tip.k] == true
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
ns = [101, 101, 201]#mesh size
steadyhead = 0e0 #meters
width = 1000 #meters
heights = [200, 100, 200]
mins = [0, 0, 0] #meters
maxs = [width, width, sum(heights)] #meters
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid3d(mins, maxs, ns)

masks = [zeros(Bool, prod(ns)) for i = 1:3]
for i = 1:size(coords, 2)
    if coords[3, i] > heights[1] + heights[2]
        masks[3][i] = true
    elseif coords[3, i] > heights[1]
        masks[2][i] = true
    else
        masks[1][i] = true
    end
end
masks = map(x->reshape(x, reverse(ns)...), masks)

#setup the fracture network
material = FractureMaker.Material(reverse(ns)...) #Logical array size equal to total number of cell
true_ind = findall(masks[2])
true_ks=[]
for i=1:length(true_ind)
    global true_ks=push!(true_ks,true_ind[i][1])
end
true_ks=unique(true_ks)
num_horizontal_fractures = sum(heights) ÷ 30 #put a horizontal fracture every 30 meters or so
tips = Array{FractureMaker.FractureTip}(undef, 2 * num_horizontal_fractures)
for frac = 1:num_horizontal_fractures
    i = rand(true_ks)
    j = rand(1:ns[2])
    k = rand(1:ns[1])
    tips[2 * frac - 1] = FractureMaker.FractureTip(i, j, k, 0, 1, 0, true)
    tips[2 * frac] = FractureMaker.FractureTip(i, j, k, 0, -1, 0,  true)
end
FractureMaker.simulate(material, tips, 9 * ns[1] ÷ 20)#the time steps let the fracture grow to be at most a little less than the width of the domain
horizontal_fracture_mask = copy(material.isfractured) .* masks[2]

num_vertical_fractures = width ÷ 10#put a vertical fracture every 10 meters or so
tips = Array{FractureMaker.FractureTip}(undef, 2 * num_vertical_fractures)
for frac = 1:num_vertical_fractures
    i = rand(true_ks)
    j = rand(1:ns[2])
    k = rand(1:ns[1])
    tips[2 * frac - 1] = FractureMaker.FractureTip(i, j, k, 1, 0, 0, true)
    tips[2 * frac] = FractureMaker.FractureTip(i, j, k, -1, 0, 0, true)
end
FractureMaker.simulate(material, tips, ns[3] * heights[2] ÷ (3 * sum(heights)))
vertical_fracture_mask = copy(material.isfractured) .* masks[2] - horizontal_fracture_mask

horizontal_fracture_mask_plot=reshape(horizontal_fracture_mask, prod(ns))
vertical_fracture_mask_plot=reshape(vertical_fracture_mask, prod(ns))

# Set up the eigenvector parameterization of the geostatistical log-permeability field
num_eigenvectors = 200
sigma = 1.0
lambda = 50
mean_log_conductivities = [log(1e-3), log(1e-8), log(1e-3)] #log(1e-4 [m/s]) -- similar to a pretty porous oil reservoir
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
parameterization = copy(grf.data.eigenfunc)
sigmas = copy(grf.data.eigenval)

logKs = zeros(reverse(ns)...)
parameterization3D=zeros(ns[3],ns[1]*ns[2],num_eigenvectors)
sigmas3D=zeros(ns[3],num_eigenvectors)
logKs2d = GaussianRandomFields.sample(grf)'#generate a random realization of the log-conductivity field
for i = 1:ns[3]#copy the 2d field to each of the 3d layers
    if i < true_ks[1]
        j = 1
    elseif i > true_ks[end]
        j = 3
    else
        j = 2
    end
    t = view(parameterization3D, i, :,:)
    t .= parameterization
    m = view(sigmas3D, i, :)
    m .= sigmas
    v = view(logKs, i, :, :)
	v .= mean_log_conductivities[j].+logKs2d
end

#play with these four numbers to make the fractures more or less likely to enable lots of flow from the lower reservoir to the upper reservoir
horizontal_fracture_mean_conductivity = log(1e-10)
vertical_fracture_mean_conductivity = log(1e-10)
horizontal_fracture_sigma_conductivity = 10
vertical_fracture_sigma_conductivity = 10
function x2logKs(x)
    logKs = [log.(exp.(sum(masks[j][i,:,:]' .* reshape(parameterization3D[i,:,:] * (sigmas3D[i,:] .* x[(j - 1) * num_eigenvectors + 1:j * num_eigenvectors]) .+ mean_log_conductivities[j], ns[1:2]...) for j = 1:3)') + 
                    horizontal_fracture_mask[i,:,:] * exp(horizontal_fracture_mean_conductivity + horizontal_fracture_sigma_conductivity * x[end - 1]) + 
                    vertical_fracture_mask[i,:,:] * exp(vertical_fracture_mean_conductivity + vertical_fracture_sigma_conductivity * x[end])) for i in 1:ns[3]]
     logKs=cat(logKs..., dims=3)  # concatenate along the first dimension
    return logKs = permutedims(logKs, (3, 2, 1))
end
logKs = x2logKs(randn(3 * num_eigenvectors + 2))
if plotstuff
    #plot the log-conductivity
    fig, ax = PyPlot.subplots()
    img = ax.imshow(logKs[:, 50, :], origin="lower")
    ax.title.set_text("Conductivity Field with fractures")
    fig.colorbar(img)
    display(fig)
    println()
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

injection_node, critical_point_node = get_nodes_near(coords, [[width / 2,width / 2, 4 * heights[1] / 5], [width / 2, width / 2,  heights[1] + heights[2] + 0.1 * heights[3]]])
injection_nodes = [injection_node]

# Set the Qs for the whole field to zero, then set it to the injection rate divided by the number of injection nodes at the injection nodes
Qs = zeros(size(coords, 2))
Qinj = 0.031688 #injection rate [m^3/s] (1 MMT water/yr)
Qs[injection_nodes] .= Qinj / length(injection_nodes)

logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
function solveforh(logKs)
    #@show length(logKs)
    @assert length(logKs) == length(Qs)
    Ks_neighbors = logKs2Ks_neighbors(logKs)
    result = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs; linear_solver=DPFEHM.amg_solver)
    return result
end

solveforheigs(x) = solveforh(x2logKs(x))
# Set up the functions that compute the pressure at the critical point and the gradient of the pressure at the critical point with respect to the eigenvector coefficients
f(x) = solveforh(x2logKs(x))[critical_point_node]*9.807*997*1e-6 #convert from head (meters) to pressure (MPa) (for water 25 C)
grad_f(x) = Zygote.gradient(f, x)[1]

if plotstuff
    import Random
    Random.seed!(1)

    x_test = randn(3 * num_eigenvectors + 2) #the input to the function should follow an N(0, I) distribution -- the eigenvectors and eigenvalues are embedded inside f
    @show x_test[[1, end - 1, end]]
    #@show f(x_test) #compute the pressure at the critical location
    #compute the gradient of the pressure at the critical location with respect to the eigenvector coefficients (then only show the largest/smallest values so it doesn't print a huge array)
    #@show extrema(grad_f(x_test))
 
    @time P_mat=reshape(solveforh(x2logKs(x_test)), reverse(ns)...)
    @show P_mat[critical_point_node]
    fig, ax = PyPlot.subplots()
    img = ax.imshow(P_mat[:, :,50], origin="lower")
    ax.title.set_text("pressure Field")
    fig.colorbar(img)
    display(fig)
    println()
    PyPlot.close(fig)
end
