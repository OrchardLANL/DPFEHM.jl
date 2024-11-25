
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

include("twoPhase.jl")

mutable struct Fluid
    vw::Float64
    vo::Float64
    swc::Float64
    sor::Float64
end

ns = [64 64]#number of nodes on the grid
mins = [0, 0];  maxs = [1-(1/64), 1-(1/64)]#size of the domain, in meters
coords, neighbors, areasoverlengths, volumes=DPFEHM.regulargrid2d(mins, maxs, ns, 1.0);#build the grid
dirichleths = zeros(size(coords, 2))
dirichletnodes=[]
specificstorage = fill(0.1, size(coords, 2))  
Qs = zeros(size(coords, 2))
Qs[[1 end]].=[1 -1];
h0 = zeros(size(coords, 2))
fluid=Fluid(1.0, 1.0, 0.0, 0.0)
S0=zeros(size(coords, 2))
nt = 2;  dt = 0.7/25;
CriticalPoint=2048
K = ones(size(coords, 2));
everystep=false # output all the time steps

function FindGrad(K)
    args=h0, S0, K, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
    P, S= solveTwoPhase(args...)
    return P[CriticalPoint]
end

@time grad_Ks= Zygote.gradient(FindGrad,K)[1]
@show grad_Ks[CriticalPoint]
using Test
function checkgradientquickly(f, x0, gradf, n; delta::Float64=1e-8, kwargs...)
	indicestocheck = sort(collect(1:length(x0)), by=i->abs(gradf[i]), rev=true)[1:n]
	f0 = f(x0)
	for i in indicestocheck
		x = copy(x0)
		x[i] += delta
		fval = f(x)
		grad_f_i = (fval - f0) / delta
		@test isapprox(gradf[i], grad_f_i; kwargs...)
        @show isapprox(gradf[i], grad_f_i; kwargs...)
	end
end
checkgradientquickly(FindGrad, K, grad_Ks, 3; delta=1e-8, rtol=1e-1)
