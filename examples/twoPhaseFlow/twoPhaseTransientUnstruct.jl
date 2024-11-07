
import DPFEHM
import PyPlot
import Zygote
import GaussianRandomFields
import Optim
import NonlinearEquations
import ChainRulesCore
import SparseArrays
using LinearAlgebra
using AlgebraicMultigrid

# include("twoPhase.jl")

mutable struct Fluid
    vw::Float64
    vo::Float64
    swc::Float64
    sor::Float64
end

ns = [64 64]#number of nodes on the grid

mins = [0, 0];  maxs = [1-(1/64), 1-(1/64)]#size of the domain, in meters
coords_o, neighbors, areasoverlengths, volumes=DPFEHM.regulargrid2d(mins, maxs, ns, 1.0);#build the grid
arr=[]
for i=1:size(coords_o,2)
    # @show coords_o[1,i]>(maxs[1]/2) && coords_o[2,i]>maxs[2]/2
    if (coords_o[1,i]>(maxs[1]/2) && coords_o[2,i]>maxs[2]/2)
        push!(arr,i)
    end
end
arr=sort(arr)
neighbor2rmv=[]
for (i,(node_a,node_b)) in enumerate(neighbors)
    if node_a in arr || node_b in arr
        push!(neighbor2rmv, i)
    end
end
@show size(coords,2)
@show size(neighbors)

neighbors_n = Array{Pair{Int, Int}}(undef, size(neighbors))

for (i,(node_a,node_b)) in enumerate(neighbors)
    
    if node_a>(ns[1]*(ns[2]/2)+1)
        j=floor(node_a/ns[2])-ns[2]/2
        new_n1=node_a-(ns[2]/2)*j
    else
        new_n1=node_a
    end
    if node_b>(ns[1]*(ns[2]/2)+1)
        j=floor(node_b/ns[2])-ns[2]/2
        new_n2=node_b-(ns[2]/2)*j
    else
        new_n2=node_b
    end
    neighbors_n[i]=new_n1=>new_n2
end

# @show neighbors[1:32]
neighbors=neighbors_n
# Create a logical index array for rows to keep
col_to_keep = trues(size(coords_o, 2))
col_to_keep[arr] .= false
coords = coords_o[:,col_to_keep]
deleteat!(neighbors, neighbor2rmv)
deleteat!(areasoverlengths, neighbor2rmv)
deleteat!(volumes, arr)


dirichleths = zeros(size(coords, 2))
dirichletnodes=[]
specificstorage = fill(0.1, size(coords, 2))  

Qs = zeros(size(coords,2))
Qs[1]=1;
Qs[32*64]=-1;
h0 = zeros(size(coords, 2))
fluid=Fluid(1.0, 1.0, 0.0, 0.0)
S0=zeros(size(coords, 2))
nt = 25;  dt = 0.7/25;
CriticalPoint=2048

K = ones(size(coords, 2))

everystep=true


args=h0, S0, K, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
P, S= DPFEHM.solveTwoPhase(args...)

# everystep=false
# function FindGrad(K)
#     args=h0, S0, K, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
#     P, S= solveTwoPhase(args...)
#     return P[CriticalPoint]
# end
# @time grad_Ks= Zygote.gradient(FindGrad,K)[1]
# @show grad_Ks[CriticalPoint]
# using Test
# function checkgradientquickly(f, x0, gradf, n; delta::Float64=1e-8, kwargs...)
# 	indicestocheck = sort(collect(1:length(x0)), by=i->abs(gradf[i]), rev=true)[1:n]
# 	f0 = f(x0)
# 	for i in indicestocheck
# 		x = copy(x0)
# 		x[i] += delta
# 		fval = f(x)
# 		grad_f_i = (fval - f0) / delta
# 		@test isapprox(gradf[i], grad_f_i; kwargs...)
#         @show isapprox(gradf[i], grad_f_i; kwargs...)
# 	end
# end
# checkgradientquickly(FindGrad, K, grad_Ks, 3; delta=1e-8, rtol=1e-1)


times = 1:1:nt  # Time steps from t=0 to t=1 with a step of 0.1
x=coords_o[1,:]
y=coords_o[2,:]
fig, axs = PyPlot.subplots()
for t=1:nt
    # Compute solution at time t
    sat=[]
    j=1
    for i=1:size(coords_o,2)
        if (coords_o[1,i]>(maxs[1]/2) && coords_o[2,i]>maxs[2]/2) #For plotting purpose only
            push!(sat,10000)
        else
            push!(sat,S[t][j])
            j=j+1
        end
    end
    # Create a filename with the time step index
    axs.imshow(reshape(sat,ns[2],ns[1]), vmin=0, vmax=1.2, origin="lower",cmap="jet")
    axs.set_aspect("equal")
    axs.set_title("Saturation")
    display(fig)
    println()

end

