#HR edit
import PyPlot
import GaussianRandomFields
import Random
import DPFEHM


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
nt = 25;  dt = 0.7/25;


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

K=reshape(exp.(logKs2d),size(coords, 2))


everystep=true # output all the time steps
args=h0, S0, K, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
P, S= DPFEHM.solvetwophase(args...)
fig, axs = PyPlot.subplots()
for t=1:nt
    # Create a filename with the time step index
    img=axs.imshow(reshape(S[t],ns[1],ns[2]), vmin=0, vmax=1, origin="lower",cmap="jet",interpolation="bicubic")
    axs.set_aspect("equal")
	axs[:tick_params](axis="both", which="major", labelsize=14)
    display(fig)
    println()

end
