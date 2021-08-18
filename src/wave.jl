# 2D acoustic wave propagation with Clayton-Enguist absorbing boundary conditions
# 4th order in space, 1st order in time
@NonlinearEquations.equations exclude=(nz, nx) function wave2diter(c, f, u, uim1, uim2, nz, nx) # 1st order
    NonlinearEquations.setnumequations(nz*nx)

    ## TOP ROW #########################
    iz = 1
    for ix=2:nx-1
        v1 = c[linearIndex(1, ix)] * dt/dz
        v2 = c[linearIndex(1, ix)] * dt/dx
        diff1 = v1*(-(uim1[linearIndex(2, ix)] - uim1[linearIndex(1, ix)] - uim2[linearIndex(2, ix)] - uim2[linearIndex(1, ix)]))
        diff2 = 0.5*v2^2*(uim1[linearIndex(1, ix-1)] -2*uim1[linearIndex(1, ix)] + uim1[linearIndex(1, ix+1)])
        NonlinearEquations.addterm((ix-1)*nz+iz, u[linearIndex(1, ix)] - 2*uim1[linearIndex(1, ix)] -diff1 - diff2)

    end

    ## SECOND FROM TOP ROW #########################
    # second to top row (x, not corners)
    iz = 2
    for ix=3:nx-2
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(-uim1[linearIndex(2, ix-2)] +16*uim1[linearIndex(2, ix-1)] + -30*uim1[linearIndex(2, ix)] +16*uim1[linearIndex(2, ix+1)] - uim1[linearIndex(2, ix+2)])/dx^2/12*c[linearIndex(2, ix)]^2)
    end
    # second from top row (z)
    for ix=1:nx
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(uim1[linearIndex(1, ix)] - 2*uim1[linearIndex(2, ix)] + uim1[linearIndex(3, ix)])/dz^2*c[linearIndex(2, ix)]^2)
    end

    ## BOTTOM ROW #########################
    for ix=2:nx-1
        v1 = c[linearIndex(nz, ix)] * dt/dz
        v2 = c[linearIndex(nz, ix)] * dt/dx
        diff1 = v1*(-(uim1[linearIndex(nz, ix)] - uim1[linearIndex(nz-1, ix)] - uim2[linearIndex(nz, ix)] - uim2[linearIndex(nz-1, ix)]))
        diff2 = 0.5*v2^2*(uim1[linearIndex(nz, ix-1)] -2*uim1[linearIndex(nz, ix)] + uim1[linearIndex(nz, ix+1)])
        NonlinearEquations.addterm((ix-1)*nz+iz, u[linearIndex(nz, ix)] - 2*uim1[linearIndex(nz, ix)] -diff1 - diff2)

    end

    ## SECOND FROM BOTTOM ROW #########################
    # second to bottom row (x, not corners)
    iz = nz - 1
    for ix=3:nx-2
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(-uim1[linearIndex(nz-1, ix-2)] +16*uim1[linearIndex(nz-1, ix-1)] + -30*uim1[linearIndex(nz-1, ix)] +16*uim1[linearIndex(nz-1, ix+1)] - uim1[linearIndex(nz-1, ix+2)])/dx^2/12*c[linearIndex(nz-1, ix)]^2)
    end
    # second from bottom row (z)
    for ix=1:nx
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(uim1[linearIndex(nz, ix)] - 2*uim1[linearIndex(nz-1, ix)] + uim1[linearIndex(nz-2, ix)])/dz^2*c[linearIndex(nz-1, ix)]^2)
    end

    # LEFT COLUMN ####################
    ix = 1
    for iz=2:nz-1
        v1 = c[linearIndex(iz, 1)] * dt/dz
        v2 = c[linearIndex(iz, 1)] * dt/dx
        diff1 = 0.5*v1^2*(uim1[linearIndex(iz-1, 1)] -2*uim1[linearIndex(iz, 1)] + uim1[linearIndex(iz+1, 1)])
        diff2 = v2*((uim1[linearIndex(iz, 2)] - uim1[linearIndex(iz, 1)] - uim2[linearIndex(iz, 2)] - uim2[linearIndex(iz, 1)]))
        NonlinearEquations.addterm((ix-1)*nz+iz, u[linearIndex(iz, 1)] - 2*uim1[linearIndex(iz, 1)] -diff1 - diff2)

    end

    # SECOND FROM LEFT COLUMN ####################
    # second to left edge (x)
    ix = 2
    for iz=1:nz
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(uim1[linearIndex(iz, 1)] - 2*uim1[linearIndex(iz, 2)] + uim1[linearIndex(iz, 3)])/dx^2*c[linearIndex(iz, 2)]^2)
    end

    # second to left edge (z, not corners)
    for iz=3:nz-2
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(-uim1[linearIndex(iz-2, 2)] +16*uim1[linearIndex(iz-1, 2)] + -30*uim1[linearIndex(iz, 2)] +16*uim1[linearIndex(iz+1, 2)] - uim1[linearIndex(iz+2, 2)])/dz^2/12*c[linearIndex(iz, 2)]^2)
    end

    # RIGHT COLUMN ####################
    ix = nx
    for iz=2:nz-1
        v1 = c[linearIndex(iz, nx)] * dt/dz
        v2 = c[linearIndex(iz, nx)] * dt/dx
        diff1 = 0.5*v1^2*(uim1[linearIndex(iz-1, nx)] -2*uim1[linearIndex(iz, nx)] + uim1[linearIndex(iz+1, nx)])
        diff2 = v2*(-(uim1[linearIndex(iz, nx)] - uim1[linearIndex(iz, nx-1)] - uim2[linearIndex(iz, nx)] - uim2[linearIndex(iz, nx-1)]))
        NonlinearEquations.addterm((ix-1)*nz+iz, u[linearIndex(iz, nx)] - 2*uim1[linearIndex(iz, nx)] -diff1 - diff2)

    end

    # SECOND FROM RIGHT COLUMN ####################
    # second to right edge (x)
    ix = nx - 1
    for iz=1:nz
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(uim1[linearIndex(iz, nx)] - 2*uim1[linearIndex(iz, nx-1)] + uim1[linearIndex(iz, nx-2)])/dx^2*c[linearIndex(iz, nx-1)]^2)
    end

    # second to right edge (z, not corners)
    for iz=3:nz-2
        NonlinearEquations.addterm((ix-1)*nz + iz, -1*(-uim1[linearIndex(iz-2, nx-1)] +16*uim1[linearIndex(iz-1, nx-1)] + -30*uim1[linearIndex(iz, nx-1)] +16*uim1[linearIndex(iz+1, nx-1)] - uim1[linearIndex(iz+2, nx-1)])/dz^2/12*c[linearIndex(iz, nx-1)]^2)
    end

    # CENTER ####################
    for iz=3:nz-2
        for ix=3:nx-2
            NonlinearEquations.addterm((ix-1)*nz + iz, -1*(-uim1[linearIndex(iz-2, ix)] +16*uim1[linearIndex(iz-1, ix)] + -30*uim1[linearIndex(iz, ix)] +16*uim1[linearIndex(iz+1, ix)] - uim1[linearIndex(iz+2, ix)])/(dz^2*12)*c[linearIndex(iz, ix)]^2) # z
            NonlinearEquations.addterm((ix-1)*nz + iz, -1*(-uim1[linearIndex(iz, ix-2)] +16*uim1[linearIndex(iz, ix-1)] + -30*uim1[linearIndex(iz, ix)] +16*uim1[linearIndex(iz, ix+1)] - uim1[linearIndex(iz, ix+2)])/(dx^2*12)*c[linearIndex(iz, ix)]^2) # x
        end
    end

    # TIME DERIVATIVE AND SOURCE ####################
    for iz=1:nz
        for ix=1:nx
            # forcing term
            NonlinearEquations.addterm((ix-1)*nz + iz, f[linearIndex(iz, ix)])

            # 1st order time derivative
            NonlinearEquations.addterm((ix-1)*nz + iz, (u[linearIndex(iz, ix)] - 2*uim1[linearIndex(iz, ix)] + uim2[linearIndex(iz, ix)])/dt^2 )
        end
    end
end

# forward model one iteration
function uoneiter(ui, uim1, uim2, c, f) # 1st order, one iteration of wave propagation
    return ui - dt^2 * wave2diter_residuals(c, f, ui, uim1, uim2, nz, nx)
end

# forward model multiple iterations
function getuIters(cval, f, nz, nx, nt)
    u = zeros(nz*nx, nt + 2)
    for i=3:nt + 2
        u[:, i] = uoneiter(zeros(size(u, 1)), u[:, i - 1], u[:, i - 2], cval, f[:, i - 2])
        if mod(i, 10) == 0
           @printf("Iteration %i\n", i)
        end
    end
    return u
end

# pullback for one iteration
function make_wave2diter_pullback(args...)
    function wave2diter_pullback(delta)
		retval = (ChainRulesCore.NO_FIELDS,#function
                @ChainRulesCore.thunk(wave2diter_c(args...)' * delta),#c
				@ChainRulesCore.thunk(wave2diter_f(args...)' * delta),#f
				@ChainRulesCore.thunk(wave2diter_u(args...)' * delta),#u
				@ChainRulesCore.thunk(wave2diter_uim1(args...)' * delta),#uim1
				@ChainRulesCore.thunk(wave2diter_uim2(args...)' * delta),#uim2
                ChainRulesCore.NO_FIELDS,#nz
                ChainRulesCore.NO_FIELDS)#nx
        return retval
    end
    return wave2diter_pullback
end

# rrule for one iteration
function ChainRulesCore.rrule(::typeof(wave2diter_residuals), c, f, u, uim1, uim2, nz, nx)
    args = (c, f, u, uim1, uim2, nz, nx)
    residuals = wave2diter_residuals(args...)
    pullback = make_wave2diter_pullback(args...)
    return residuals, pullback
end

# rrule rule for multiple iterations
function ChainRulesCore.rrule(::typeof(getuIters), c, f, nz, nx, nt)
    u = getuIters(c, f, nz, nx, nt)
    return u, delta->begin
        ds = []
        #last iteration
        _, pb = Zygote.pullback(uoneiter, zeros(size(u, 1)), u[:, size(u, 2) - 1], u[:, size(u, 2) - 2], c, f[:, end])
        d1, d2, d3, d4, d5 = pb(delta[:, size(u, 2)])
        push!(ds, (d1, d2, d3))
        dc = d4
        df = hcat(d5)
        #second to last iteration
        _, pb = Zygote.pullback(uoneiter, zeros(size(u, 1)), u[:, size(u, 2) - 2], u[:, size(u, 2) - 3], c, f[:, end - 1])
        d1, d2, d3, d4, d5 = pb(delta[:, size(u, 2) - 1] + ds[1][2])
        push!(ds, (d1, d2, d3))
        dc += d4
        df = hcat(d5, df)
        #now do the general iteration
        for i = size(u, 2) - 2:-1:3
            _, pb = Zygote.pullback(uoneiter, zeros(size(u, 1)), u[:, i - 1], u[:, i - 2], c, f[:, i - 2])
            d1, d2, d3, d4, d5 = pb(delta[:, i] + ds[2][2] + ds[1][3])
            dc += d4
            df = hcat(d5, df)
            push!(ds, (d1, d2, d3))
            ds = ds[2:end]
        end
        return (ChainRulesCore.NO_FIELDS,#getuIters
                dc,
                df,
                ChainRulesCore.NO_FIELDS,#nz
                ChainRulesCore.NO_FIELDS,#nx
                ChainRulesCore.NO_FIELDS)#nt
    end
end
