# solving the wave equation
# \frac{\partial ^2 u}{\partial t ^2} - c^2 [ \frac{\partial ^2 u}{\partial z^2} + \frac{\partial ^2 u}{\partial x^2}] + f = 0

import DPFEHM
#using Printf 
using Zygote
import PyPlot
## FUNCTIONS ####################

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
    return l2norm
end

# plot shot record
function plotShot(r, nt, dt, nr)
    w, h = figaspect(1.6)
    figure(figsize=[1.5*w, 1.5*h])
    boundary = 0.95*maximum(abs.(r))
    imshow(r', extent=[1, nr, dt*nt, 0], vmin=-boundary, vmax=boundary, aspect="auto", cmap="seismic")
    cb = colorbar()
    cb.set_label("Amplitude")
    ylabel("Time (s)")
    xlabel("Receiver number")
    title("Shot record")
end


## PROBLEM SETUP ####################
# NOTE: z is depth, x is horizontal distance. Indexing is (z, x)
# time steps
nt = 500    # number of time steps (unitless)
dt = 0.001  # how long each time step is (seconds)

# spatial steps
nz = 100    # number of spatial steps in z (unitless)
nx = 100    # number of spatial steps in x (unitless)
dz = 5      # how long each spatial step is in z (meters)
dx = 5      # how long each spatial step is in x (meters)

#z = range(0, length=nz, step=dz)   # can be helpful for plotting
#x = range(0, length=nx, step=dx)   # can be helpful for plotting

# wavelet
t = range(0, stop=nt*dt, length=nt)
wave = 1000 .*ricker(45, t; Ã=0.06)

# create forcing term (ricker wavelet)
f = zeros(nz, nx, nt)
halfx = floor(Int, nx/2)
halfz = floor(Int, nz/2)
f[1, halfx, :] = wave
f = reshape(f, nz * nx, nt)

# get starting (constant background) velocity model
c = 2000 .*ones(nz, nx) # m/s
c = reshape(c, nz * nx)

# truth velocity model (scatterer in center of domain)
ctrue = 2000 .*ones(nz, nx) # m/s
#ctrue[halfz, halfx] = 2800 # m/s
sz = 4 # scatterer size
ctrue[halfz-sz:halfz+sz, halfx-sz:halfx+sz] .= 2500 # m/s
#ctrue[halfz-sz-25:halfz+sz-25, halfx-sz:halfx+sz] .= 3200 # m/s
ctrue = reshape(ctrue, nz * nx)

# receiver locations along surface
xloc = Array(1:nx)
zloc = 5 .*ones(size(xloc))
ridx = floor.(Int, DPFEHM.linearIndex.(nz.*ones(size(xloc)), zloc, xloc))

# get wavefield in background medium (can comment out)
@time u = DPFEHM.getuIters(c, f, nz, nx, nt, dz, dx, dt)
fig, axs = PyPlot.subplots(2, 13, figsize=(30, 5))
axs = permutedims(axs, [2, 1])
j = 1
#for i = 2:10:size(u, 2)
for i = 2:20:size(u, 2)
    global j
    axs[j].imshow(reshape(u[:, i], nz, nx), cmap="seismic")
    j += 1
end
fig.tight_layout()


# get wavefield in truth velocity model
@time utrue = DPFEHM.getuIters(ctrue, f, nz, nx, nt, dz, dx, dt)
fig, axs = PyPlot.subplots(2, 13, figsize=(30, 5))
axs = permutedims(axs, [2, 1])
j = 1
#for i = 2:10:size(u, 2)
for i = 2:20:size(utrue, 2)
    global j
    axs[j].imshow(reshape(utrue[:, i], nz, nx), cmap="seismic")
    j += 1
end
fig.tight_layout()

# shot record (data recorded at receivers)
shotRecord = utrue[ridx, :] # get the shot record

# gradient functions
g3(c, f) = getNorm(c, f, nz, nx, nt, dt, ridx, shotRecord)
g3(c) = getNorm(c, f, nz, nx, nt, dt, ridx, shotRecord)

# Full waveform inversion (with gradient descent)
niter=10                            # number of iterations
csave = zeros(size(c, 1), niter)    # save c at every iteration
csave[:, 1] = c                     # initial c at first iteration
stepLength=1e10 .*ones(niter)       # step length 
l2save = zeros(niter)               # save loss function at every iteration

# update model
#for i = 1:2 # for one iteration (aka RTM)
for i = 1:niter-1
    l2save[i] = g3(csave[:, i])
    @show i
    grad = Zygote.gradient(g3, csave[:, i])
    csave[:, i+1] = csave[:, i] .- stepLength[i].*reshape(grad[1], nz*nx) 
    #@printf("Done with iteration %i\n", i)
end
