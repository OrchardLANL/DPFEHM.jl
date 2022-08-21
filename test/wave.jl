using Test
import DPFEHM
import Zygote

function checkgradientquickly(f, x0, gradf, n; delta::Float64=1e-8, kwargs...)
	indicestocheck = sort(collect(1:length(x0)), by=i->abs(gradf[i]), rev=true)[1:n]
	#indicestocheck = [indicestocheck; rand(1:length(x0), n)]
	f0 = f(x0)
	for i in indicestocheck
		x = copy(x0)
		x[i] += delta
		fval = f(x)
		grad_f_i = (fval - f0) / delta
		@test isapprox(gradf[i], grad_f_i; kwargs...)
	end
end

# generate source wavelet (Ricker wavelet)
function ricker(f, t; Ã=0)  # generate ricker wavelet
    return (1 .- 2 .* pi.^2 .* f.^2 .* (t.-Ã).^2) .* exp.(-pi.^2 .* f.^2 .* (t.-Ã).^2)
end

# get norm to generate objective function
function getNorm(c, f, nz, nx, nt, dt, ridx, ur)
    usave = DPFEHM.getuIters(c, f, nz, nx, nt, dz, dx, dt)
    l2norm = 0
    for i=1:length(ridx)
        l2norm += sum((ur[i, :] .- usave[ridx[i], :]).^2)
    end
    c_matrix = reshape(c, nz, nx)
    grad_reg = sum((c_matrix[2:end, :] - c_matrix[1:end - 1, :]) .^ 2) + sum((c_matrix[:, 2:end] - c_matrix[:, 1:end - 1]) .^ 2)
    cbin = (c .- 2000) ./ 500
    clow = c .< 2250
    chigh = c .>= 2250
    bin_reg = (sum((cbin .* clow) .^ 2) + sum((cbin .* (chigh .- 1)) .^ 2)) / (4 * length(c))
    return l2norm / 1e-6 + grad_reg / (1e4 * 500 ^ 2) + 10 * bin_reg
end

# NOTE: z is depth, x is horizontal distance. Indexing is (z, x)
# time steps
nt = 100    # number of time steps (unitless)
dt = 0.010  # how long each time step is (seconds)
# spatial steps
nz = 100    # number of spatial steps in z (unitless)
nx = 100    # number of spatial steps in x (unitless)
dz = 5      # how long each spatial step is in z (meters)
dx = 5      # how long each spatial step is in x (meters)
# wavelet
t = range(0, stop=nt*dt, length=nt)
wave = 1000 .*ricker(45, t; Ã=0.06)
# create forcing term (ricker wavelet)
f_source = zeros(nz, nx, nt)
halfx = floor(Int, nx/2)
halfz = floor(Int, nz/2)
f_source[1, halfx, :] = wave
f_source = reshape(f_source, nz * nx, nt)
# get starting (constant background) velocity model
c = 2000 .*ones(nz, nx) # m/s
c = reshape(c, nz * nx)
# truth velocity model (scatterer in center of domain)
ctrue = 2000 .*ones(nz, nx) # m/s
sz = 10 # scatterer size
ctrue[halfz-sz:halfz+sz, halfx-sz:halfx+sz] .= 2500 # m/s
ctrue = reshape(ctrue, nz * nx)
# receiver locations along surface
xloc = Array(1:nx)
zloc = 5 .*ones(size(xloc))
ridx = floor.(Int, DPFEHM.linearIndex.(nz.*ones(size(xloc)), zloc, xloc))
# get wavefield in truth velocity model
utrue = DPFEHM.getuIters(ctrue, f_source, nz, nx, nt, dz, dx, dt)
# shot record (data recorded at receivers)
shotRecord = utrue[ridx, :] # get the shot record
# gradient functions
g3(c) = getNorm(c, f_source, nz, nx, nt, dt, ridx, shotRecord)
gradg3 = Zygote.gradient(g3, ctrue)[1]
#check that it is giving us the right gradients
checkgradientquickly(g3, ctrue, gradg3, 3; atol=1e-8, rtol=1e-4)
