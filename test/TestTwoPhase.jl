using SparseArrays
import PyPlot
using IterativeSolvers
using AlgebraicMultigrid
function cg_solver(A, b; kwargs...)
    ##for amg solver
	# ml = AlgebraicMultigrid.ruge_stuben(A)
	# # hfree = AlgebraicMultigrid._solve(ml, b; kwargs...)
    # hfree =IterativeSolvers.cg(A, b; kwargs...)
     hfree =A\b
	return hfree
end

mutable struct Fluid
	vw::Float64
	vo::Float64
	swc::Float64
	sor::Float64
end

mutable struct Grid{S,T,R}
	Nx::S
	Ny::S
	Nz::S
	hx::T
	hy::T
	hz::T
	V::T
	K::R
	por::R
end

mutable struct Velocity{T}
	x::T
	y::T
	z::T
end

function TPFA(grid,K,q,kwargs...)
	# Compute transmissibilities by harmonic averaging.
	Nx=grid.Nx; Ny=grid.Ny; Nz=grid.Nz; N=Nx*Ny*Nz; hx=grid.hx; hy=grid.hy; hz=grid.hz;
	L = K.^(-1);
	tx = 2*hy*hz/hx; TX = zeros(Nx+1,Ny,Nz);
	ty = 2*hx*hz/hy; TY = zeros(Nx,Ny+1,Nz);
	tz = 2*hx*hy/hz; TZ = zeros(Nx,Ny,Nz+1);
	TX[2:Nx,:,:] = tx./(L[1,1:Nx-1,:,:]+L[1,2:Nx ,:,:]);
	TY[:,2:Ny,:] = ty./(L[2,:,1: Ny-1,:]+L[2,:,2:Ny,:]);
	TZ[:,:,2:Nz] = tz./(L[3,:,:,1:Nz-1]+L[3,:,:,2:Nz]);
	# Assemble TPFA discretization matrix.
	x1 = reshape(TX[1:Nx,:,:],N); x2 = reshape(TX[2:Nx+1,:,:],N);
	y1 = reshape(TY[:,1:Ny,:],N); y2 = reshape(TY[:,2:Ny+1,:],N);
	z1 = reshape(TZ[:,:,1:Nz],N); z2 = reshape(TZ[:,:,2:Nz+1],N);
	DiagVecs = [-z2[1:end - Nx * Ny],-y2[1:end - Nx],-x2[1:end - 1],x1+x2+y1+y2+z1+z2,-x1[2:end],-y1[Nx + 1:end],-z1[Nx * Ny + 1:end]];
	DiagIndx = [-Nx*Ny,-Nx,-1,0,1,Nx,Nx*Ny];
	A = spdiagm(N, N, map((x, y)->x=>y, DiagIndx, DiagVecs)...)
	A[1,1] = A[1,1]+sum(grid.K[:,1,1]);
    u=cg_solver(A, q; kwargs...)
	P = reshape(u,Nx,Ny,Nz);
	Vx = zeros(Nx+1,Ny,Nz);
	Vy = zeros(Nx,Ny+1,Nz);
	Vz = zeros(Nx,Ny,Nz+1);
	Vx[2:Nx,:,:] =(P[1:Nx-1,:,:]-P[2:Nx,:,:]).*TX[2:Nx,:,:];
	Vy[:,2:Ny,:] = (P[:,1:Ny-1,:]-P[:,2:Ny,:]).*TY[:,2:Ny,:];
	Vz[:,:,2: Nz] = (P[:,:,1: Nz-1]-P[:,:,2:Nz]).*TZ[:,:,2:Nz];
	V = Velocity(Vx, Vy, Vz)
	return P, V, A
end

N = 64
grid = Grid(N, N, 1, 1 / N, 1 / N, 1.0, 1.0 / N ^ 2, ones(3,N,N), ones(N, N, 1))
Dx = Dy = 1
q=zeros(N * N * 1);
q[[1 N * N]].=[1 -1];
P, V, A = TPFA(grid,grid.K,q);
function Pres(grid, S, fluid, q)
	# Compute K∗lambda(S)
	Mw,Mo=RelPerm(S,fluid);
	Mt=Mw+Mo;
	KM = reshape(hcat(Mt,Mt,Mt)',3,grid.Nx,grid.Ny,grid.Nz).*grid.K;
	# Compute pressure and extract fluxes
	P,V=TPFA(grid,KM,q);
	return P, V
end

function RelPerm(s,fluid)
	S = (s.-fluid.swc)/(1-fluid.swc-fluid.sor); Mw = S.^2/fluid.vw;
	Mo =(1 .- S).^2/fluid.vo;
	# Rescale saturations % Water mobility
	# Oil mobility
	dMw = 2*S/fluid.vw/(1-fluid.swc-fluid.sor);
	dMo = -2*(1 .- S)/fluid.vo/(1-fluid.swc-fluid.sor);
	return Mw, Mo, dMw, dMo
end

function Upstream(grid,S,fluid,V,q,T)
	Nx=grid.Nx; Ny=grid.Ny; Nz=grid.Nz;
	N=Nx*Ny*Nz;
	pv = grid.V .* grid.por[:];

	fi = max.(q, 0)
	XP=max.(V.x,0); XN=min.(V.x,0); YP=max.(V.y,0); YN=min.(V.y,0); ZP=max.(V.z,0); ZN=min.(V.z,0);

	Vi = XP[1:Nx,:,:]+YP[:,1:Ny,:]+ZP[:,:,1:Nz]-XN[2:Nx+1,:,:]-YN[:,2:Ny+1,:]-ZN[:,:,2:Nz+1];
	pm = minimum(pv./(Vi[:]+fi));
	cfl = ((1-fluid.swc-fluid.sor)/3)*pm;
	Nts = ceil(T/cfl);
	dtx = (T/Nts)./pv;

	A=GenA(grid,V,q);
	A=spdiagm(N, N, 0=>dtx)*A;
	fi =max.(q,0).*dtx;
	for t=1:Nts
		mw,mo=RelPerm(S,fluid); fw = mw./(mw+mo);
		S = S+(A*fw+fi);
	end
	return S
end

function GenA(grid,V,q)
	Nx=grid.Nx; Ny=grid.Ny; Nz=grid.Nz; N=Nx*Ny*Nz;
	N=Nx*Ny*Nz;
	fp=min.(q,0);
	XN=min.(V.x,0); x1=reshape(XN[1:Nx,:,:],N);
	YN=min.(V.y,0); y1=reshape(YN[:,1:Ny,:],N);
	ZN=min.(V.z,0); z1=reshape(ZN[:,:,1:Nz],N);
	XP=max.(V.x,0); x2=reshape(XP[2:Nx+1,:,:],N);
	YP=max.(V.y,0); y2=reshape(YP[:,2:Ny+1,:],N);
	ZP=max.(V.z,0); z2=reshape(ZP[:,:,2:Nz+1],N);
	DiagVecs=[z2[1:end - Nx * Ny],y2[1:end - Nx],x2[1:end - 1],fp+x1-x2+y1-y2+z1-z2,-x1[2:end],-y1[Nx + 1:end],-z1[Nx * Ny + 1:end]]; # diagonal vectors
	DiagIndx=[-Nx*Ny,-Nx,-1,0,1,Nx,Nx*Ny]; # diagonal index
	A = spdiagm(N, N, map((x, y)->x=>y, DiagIndx, DiagVecs)...)
	return A
end

P_data=[]
S_data=[]
fluid = Fluid(1.0, 1.0, 0.0, 0.0)
S=zeros(N * N * 1);
nt = 25; dt = 0.7/nt;
for t=1:nt
	global S
	global P
	global V
	P,V=Pres(grid,S,fluid,q);
	S=Upstream(grid,S,fluid,V,q,dt);
    push!(P_data, deepcopy(reshape(P, grid.Nx*grid.Ny)))
    push!(S_data, deepcopy(reshape(S, grid.Nx*grid.Ny)))
end

P_matlab=P_data
S_matlab=S_data


### DPFEHM 
include("twoPhase.jl")

mutable struct Fluid_dp
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
fluid=Fluid_dp(1.0, 1.0, 0.0, 0.0)
S0=zeros(size(coords, 2))
nt = 2;  dt = 0.7/25;
CriticalPoint=2048
K = ones(size(coords, 2));
everystep=false # output all the time steps

#First test the gradient
function FindGrad(K)
    args=h0, S0, K, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
    P, S= solveTwoPhase(args...)
    return P[CriticalPoint]
end
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
@time grad_Ks= Zygote.gradient(FindGrad,K)[1]
checkgradientquickly(FindGrad, K, grad_Ks, 3; delta=1e-8, rtol=1e-1)

# Test the value with reference solution 
everystep=true # output all the time steps

#First test the gradient
nt = 25; 
args=h0, S0, K, dirichleths,  dirichletnodes, Qs, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
P_dp, S_dp= solveTwoPhase(args...)

total_norm=0
fig, axs = PyPlot.subplots(1, 2, figsize=(8, 4.5))
for i=1:nt
    total_norm=total_norm+norm(P_matlab[i]-P_dp[i])+norm(S_matlab[i]-S_dp[i])
    axs[1].imshow(reshape(S_dp[i],ns[1],ns[2]), vmin=0, vmax=1, origin="lower")
	axs[1].set_aspect("equal")
	axs[2].imshow(reshape(S_matlab[i],ns[1],ns[2]), vmin=0, vmax=1, origin="lower")
	axs[2].set_aspect("equal")
	display(fig)
	println()
end
@test total_norm<1e-5
@show total_norm

P_diff=P_matlab[end].-P_dp[end]
S_diff=S_matlab[end].-S_dp[end]


#Plot Error
fig, ax = PyPlot.subplots()
contour = ax.contourf(reshape(P_diff,grid.Nx,grid.Ny), 20)
ax.set_title("Pressure")
fig.colorbar(contour, ax=ax)
display(fig)
PyPlot.close(fig)


fig, ax = PyPlot.subplots()
contour = ax.contourf(reshape(S_diff,grid.Nx,grid.Ny), 20)
ax.set_title("Saturation")
fig.colorbar(contour, ax=ax)
display(fig)
PyPlot.close(fig)