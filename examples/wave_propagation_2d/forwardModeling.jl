# solving the wave equation
# \frac{\partial ^2 u}{\partial t ^2} - c^2 [ \frac{\partial ^2 u}{\partial z^2} + \frac{\partial ^2 u}{\partial x^2}] + f = 0

# INPUT model: ctrue
# OUTPUT data: shotRecord

import DPFEHM
import Optim
import PyPlot
import Zygote

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
    c_matrix = reshape(c, nz, nx)
    grad_reg = sum((c_matrix[2:end, :] - c_matrix[1:end - 1, :]) .^ 2) + sum((c_matrix[:, 2:end] - c_matrix[:, 1:end - 1]) .^ 2)
    cbin = (c .- 2000) ./ 500
    clow = c .< 2250
    chigh = c .>= 2250
    bin_reg = (sum((cbin .* clow) .^ 2) + sum((cbin .* (chigh .- 1)) .^ 2)) / (4 * length(c))
    return l2norm / 1e-6 + grad_reg / (1e4 * 500 ^ 2) + 10 * bin_reg
end

# plot shot record (output data)
function plotShot(r, nt, dt, nr)
    w, h = PyPlot.figaspect(1.6)
    PyPlot.figure(figsize=[1.5*w, 1.5*h])
    boundary = 0.95*maximum(abs.(r))
    PyPlot.imshow(r', extent=[1, nr, dt*nt, 0], vmin=-boundary, vmax=boundary, aspect="auto", cmap="seismic")
    cb = PyPlot.colorbar()
    cb.set_label("Amplitude")
    PyPlot.ylabel("Time (s)")
    PyPlot.xlabel("Receiver number")
    PyPlot.title("Shot record")
end

# plot velocity model (input model)
function plotVelocity(vel, nx, nz, dx, dz, dt, xloc, zloc, sx, sz, pad)
    indz = (Array(0:nz-1).*dz)[zloc]
    indx = (Array((-dx*nx/2):dx:(dx*nx/2)))[xloc]

    w, h = PyPlot.figaspect(0.75 * nz/(nx-2*pad))
    PyPlot.figure(figsize=[2.0*w,2.0*h])
    PyPlot.imshow(vel[:,pad:end-pad], cmap="plasma", extent=[-dz*nz/2, dz*nz/2, dx*(nx-2*pad), 0])
    PyPlot.xlim([-dz*nz/2, dz*nz/2])
    PyPlot.xlabel("X (m)")
    PyPlot.ylabel("Z (m)")
    PyPlot.title("Velocity model")
    cb = PyPlot.colorbar()
    cb.set_label("Velocity (m/s)")
    PyPlot.scatter(indx, indz, marker=7, color="black", clip_on=false)
    PyPlot.ylim([dx*(nx-2*pad), 0])

    indz = (Array(0:nz-1).*dz)[sz]
    indx = (Array((-dx*nx/2):dx:(dx*nx/2)))[sx]

    PyPlot.scatter(indx, indz, marker="X", s=102, color="orangered", clip_on=false)
end

## PROBLEM SETUP ####################
# NOTE: z is depth, x is horizontal distance. Indexing is (z, x)
# time steps
nt = 500    # number of time steps (unitless)
dt = 0.001  # how long each time step is (seconds)

# padding for boundary conditions
pad = 50    # if wave reflected from boundary is visible in shotRecord, increase this number

# spatial steps
nz = 100            # number of spatial steps in z (unitless)
nx = 100 + 2*pad    # number of spatial steps in x (unitless)
dz = 5              # how long each spatial step is in z (meters)
dx = 5              # how long each spatial step is in x (meters)

# wavelet
t = range(0, stop=nt*dt, length=nt)
wave = 1e5 .*ricker(45, t; Ã=0.06)

# create forcing term (ricker wavelet)
f = zeros(nz, nx, nt)
halfx = floor(Int, nx/2)
halfz = floor(Int, nz/2)
sz = 1              # source z location
sx = halfx          # source x location
f[sz, sx, :] = wave
f = reshape(f, nz * nx, nt)

# create velocity model (scatterer in center of domain)
ctrue = 2000 .*ones(nz, nx) # m/s
sz = 2 # scatterer size
ctrue[halfz-sz:halfz+sz, halfx-sz:halfx+sz] .= 2800 # m/s
ctrue = reshape(ctrue, nz * nx)

# receiver locations along surface
xloc = Int.(Array(pad:nx-pad))  # padding on either side 
zloc = Int.(3 .*ones(size(xloc)))
ridx = floor.(Int, DPFEHM.linearIndex.(nz.*ones(size(xloc)), zloc, xloc))

# get wave propagation 
@time utrue = DPFEHM.getuIters(ctrue, f, nz, nx, nt, dz, dx, dt)

# shot record (data recorded at receivers)
shotRecord = utrue[ridx, :] # get the shot record

## PLOTTING ####################

# plot wave propagation
fig, axs = PyPlot.subplots(2, 13, figsize=(30, 5))
axs = permutedims(axs, [2, 1])
j = 1
for i = 2:20:size(utrue, 2)
    global j
    axs[j].imshow(reshape(utrue[:, i], nz, nx), cmap="seismic")
    j += 1
end
fig.tight_layout()
display(fig)

# plot shot record and velocity
plotShot(shotRecord, nt, dt, length(ridx))
plotVelocity(reshape(ctrue, nz, nx), nx, nz, dx, dz, dt, xloc, zloc, sx, sz, pad)
