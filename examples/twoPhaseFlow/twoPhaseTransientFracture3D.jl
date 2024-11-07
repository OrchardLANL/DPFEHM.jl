
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

ns = [50 50 5]#number of nodes on the grid
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
K = ones(reverse(ns)...);
K[:,25,15:35]=K[:,25,15:35]*100
K[:,15:35,25]=K[:,15:35,25]*100
K[:,25,25]=K[:,25,25]/100
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
P_data, S_data= DPFEHM.solveTwoPhase(args...)

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
@show S_data[end][1]
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
