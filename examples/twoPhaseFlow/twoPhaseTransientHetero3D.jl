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
import Random

 include("twoPhase.jl")

mutable struct Fluid
    vw::Float64
    vo::Float64
    swc::Float64
    sor::Float64
end

ns = [60 60 5]#number of nodes on the grid
mins = [0, 0, 0];  maxs = [2.5, 2.5, 0.25]#size of the domain, in meters
coords, neighbors, areasoverlengths, volumes=DPFEHM.regulargrid3d(mins, maxs, ns);#build the grid
dirichleths = zeros(size(coords, 2))
dirichletnodes=[]
specificstorage = fill(0.1, size(coords, 2))  
Qs = zeros(size(coords, 2))
Qs[[1 end]].=[1 -1];
h0 = zeros(size(coords, 2))
fluid=Fluid(1.0, 1.0, 0.0, 0.0)
S0=zeros(size(coords, 2))
nt = 25;  dt = 0.7/25;
CriticalPoint=ceil(size(coords,2))


logKs = zeros(reverse(ns)...)

for i = 1:ns[3]
    Random.seed!(1)
    lambda = 10.0#meters -- correlation length of log-conductivity
    sigma = 14#standard deviation of log-conductivity
    mu = -15.0#mean of log conductivity -- ~1e-4 m/s, like clean sand here https://en.wikipedia.org/wiki/Hydraulic_conductivity#/media/File:Groundwater_Freeze_and_Cherry_1979_Table_2-2.png
    cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
    x_pts = range(mins[1], maxs[1]; length=ns[1])
    y_pts = range(mins[2], maxs[2]; length=ns[2])
    num_eigenvectors = 200
    grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)
    logKs2d = mu .+ GaussianRandomFields.sample(grf)'#generate a random realization of the log-conductivity field
    #copy the 2d field to each of the 3d layers
    v = view(logKs, i, :, :)
    v .= logKs2d
end

K=exp.(logKs)
fig, ax = PyPlot.subplots()
img = ax.imshow(K[1, :, :], origin="lower")
ax.title.set_text("Conductivity Field")
fig.colorbar(img)
display(fig)
println()
PyPlot.close(fig)
K=reshape(K,size(coords,2))
everystep=true
args=h0, S0, K, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
P_data, S_data= solveTwoPhase(args...)

# Plot in Paraview
using WriteVTK
using Printf 

times = 1:1:nt 
x=reshape(coords[1,:], reverse(ns)...)
y=reshape(coords[2,:], reverse(ns)...)
z=reshape(coords[3,:], reverse(ns)...)
for t=1:nt
    # Compute solution at time t
    sat = reshape(S_data[t], reverse(ns)...)
    pres = reshape(P_data[t], reverse(ns)...)
    # Create a filename with the time step index
    filename = @sprintf("output_%03dfrac", t)
  
    # Write to a VTK file
    vtk_grid(filename, x, y, z) do vtk
        vtk["Saturation"] = sat
        vtk["Pressure"] = pres
        vtk["perm"] = K
    end
end

function write_pvd_file(filenames, times, pvd_filename="3dFracoutput.pvd")
    open(pvd_filename, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")
        println(io, "  <Collection>")
        for (filename, t) in zip(filenames, times)
            println(io, @sprintf("    <DataSet timestep=\"%f\" group=\"\" part=\"0\" file=\"%s\"/>", t, filename))
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
end

# Generate list of filenames and times
filenames = [ @sprintf("output_%03dfrac.vts", idx) for idx in 1:length(times) ]
write_pvd_file(filenames, times)
#The following code can be used to calculate gradient 
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
