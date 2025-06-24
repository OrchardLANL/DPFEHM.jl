# Source code for twophase flow (HR edit)
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
using IterativeSolvers

#solver
function cg_solver(A, b; kwargs...)
	ml = AlgebraicMultigrid.ruge_stuben(A)
	hfree = AlgebraicMultigrid._solve(ml, b; reltol=1e-14,kwargs...)
	return hfree
end

function getfreenodes(n, dirichletnodes)
    isfreenode = [i ∉ dirichletnodes for i in 1:n]
    nodei2freenodei = [isfreenode[i] ? sum(isfreenode[1:i]) : -1 for i in 1:n]
    freenodei2nodei = [i for i in 1:n if isfreenode[i]]
    return isfreenode, nodei2freenodei, freenodei2nodei
end

# Macro for the governing equations of two phase flow pressure. It calculates residuals and jacobian matrix automatically
@NonlinearEquations.equations exclude=( dirichletnodes,neighbors, areasoverlengths) function transmissivity(h,Ks_neighbors,neighbors,areasoverlengths,dirichletnodes,dirichleths,Qs)
    isfreenode, nodei2freenodei, = getfreenodes(length(Qs), dirichletnodes)
    NonlinearEquations.setnumequations(sum(isfreenode))
    tx = 2*areasoverlengths;
    for i = 1:length(Qs)
        if isfreenode[i]
            j = nodei2freenodei[i]
            NonlinearEquations.addterm(j, -Qs[i])
        end
    end
    for (i, (node_a, node_b)) in enumerate(neighbors)
        for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
            j1 = nodei2freenodei[node1]
            if isfreenode[node1] && isfreenode[node2]
                j2 = nodei2freenodei[node2]
                NonlinearEquations.addterm(j1,  (h[j1] - h[j2]) *(tx[i]*Ks_neighbors[i]))
            elseif isfreenode[node1] && !isfreenode[node2]
                NonlinearEquations.addterm(j1,  (h[j1] - dirichleths[node2]) *(tx[i]*Ks_neighbors[i]))
            end
        end
    end
end


#pressure solver
function solvepressure(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, kwargs...)
    isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(Qs), dirichletnodes)
    args = (zeros(sum(isfreenode)), Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
    A = transmissivity_h(args...)
    b= -transmissivity_residuals(args...)
    hfree=cg_solver(A, b; kwargs...)
    h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
    return h
end

# backpropgation (pressure solver)
function ChainRulesCore.rrule(::typeof(solvepressure), Ks_neighbors, neighbors,areasoverlengths,dirichletnodes,dirichleths,Qs, kwargs...)
	isfreenode, nodei2freenodei,  = getfreenodes(length(Qs), dirichletnodes)
    args_noh = (zeros(sum(isfreenode)), Ks_neighbors, neighbors, areasoverlengths,dirichletnodes,dirichleths,Qs)
    A = transmissivity_h(args_noh...)
    b= -transmissivity_residuals(args_noh...)
    hfree=cg_solver(A, b; kwargs...)
    h = map(i->isfreenode[i] ? hfree[nodei2freenodei[i]] : dirichleths[i], 1:length(Qs))
	function pullback(delta)
		args = (hfree, Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
        lambda=cg_solver(SparseArrays.SparseMatrixCSC(A'), delta[isfreenode]; kwargs...)
		trans_Ks = transmissivity_Ks_neighbors(args...)
        trans_dirichleths = transmissivity_dirichleths(args...)
		trans_Qs = transmissivity_Qs(args...)

		return (ChainRulesCore.NoTangent(),#function
				@ChainRulesCore.thunk(-(trans_Ks' * lambda)),#Ks
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#neighbors
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#areasoverlengths
				@ChainRulesCore.thunk(ChainRulesCore.NoTangent()),#dirichletnodes
				@ChainRulesCore.thunk(-(trans_dirichleths' * lambda) .+ delta .* (map(x->!x, isfreenode))),#dirichleths
				@ChainRulesCore.thunk(-(trans_Qs' * lambda)))   #Qs
    end
	return h, pullback
end

#Macro for the governing equations of two phase flow saturation. It calculates residuals and jacobian matrix automatically
@NonlinearEquations.equations  function saturationcalc(f,Qs,neighbors,P,Vn)
    dirichletnodes=[]
    isfreenode, nodei2freenodei, = getfreenodes(length(Qs), dirichletnodes)
    NonlinearEquations.setnumequations(length(Qs))
    fp=min.(Qs,0)
	for j = 1:length(Qs)
		NonlinearEquations.addterm(j, fp[j] * f[j])
	end
    for (i, (node_a, node_b)) in enumerate(neighbors) 
        for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
            j1 = nodei2freenodei[node1]
            if isfreenode[node1] && isfreenode[node2]
                j2 = nodei2freenodei[node2]     
                upwind = ((P[j2]-P[j1]) >= 0)
                if upwind
                    if Vn[i]>0
                        NonlinearEquations.addterm(j1,(f[j2])*(Vn[i]))
                    else
                        NonlinearEquations.addterm(j1,-(f[j2])*(Vn[i]))
                    end
                else
                    if Vn[i]>0
                        NonlinearEquations.addterm(j1,-(f[j1])*(Vn[i]))
                    else
                        NonlinearEquations.addterm(j1,(f[j1])*(Vn[i]))
                    end
                end
            end
        end
    end
end

# #backpropgation (saturation macro)
function ChainRulesCore.rrule(::typeof(saturationcalc_residuals),f::AbstractVector,Qs::AbstractVector,neighbors,P::AbstractVector,Vn::AbstractVector)
    dirichletnodes=[]
    R = saturationcalc_residuals(f, Qs, neighbors, P, Vn)
    isfreenode, node2free, = getfreenodes(length(Qs), dirichletnodes)
    function pullback(ΔR::AbstractVector)
        ∂f  = zeros(length(f))
        ∂Qs = zeros(length(Qs))
        ∂P  = zeros(length(P))   
        ∂Vn = zeros(length(Vn))
        # “Mass‑term”:
        fp = min.(Qs, 0.0)
        for j in 1:length(Qs)
            ∂f[j]  += fp[j] * ΔR[j]
            for j in eachindex(Qs)
                if Qs[j] < 0
                    ∂Qs[j] += f[j] * ΔR[j]
                end
            end
        end
        # — “Flux‑term”:
        for (i, (na, nb)) in enumerate(neighbors)
            j1 = node2free[na]
            j2 = node2free[nb]
            # only interior (both free) contribute
            if isfreenode[na] && isfreenode[nb]
                up = (P[j2] - P[j1]) ≥ 0
                s = Vn[i] > 0 ? 1.0 : -1.0
                # direction na→nb
                if up
                    ∂f[nb] += s * Vn[i] * ΔR[j1]
                else
                    ∂f[na] += -s * Vn[i] * ΔR[j1]
                end
                # direction nb→na
                up₂ = (P[j1] - P[j2]) ≥ 0
                if up₂
                    ∂f[na] += s * Vn[i] * ΔR[j2]
                else
                    ∂f[nb] += -s * Vn[i] * ΔR[j2]
                end
            end
        end
        # loop over each interface (edge) i
        for (i, (na, nb)) in enumerate(neighbors)
            # find the local residual‐indices for each cell
            j1 = node2free[na]
            j2 = node2free[nb]
            # skip if either cell is Dirichlet (no residual eqn)
            if isfreenode[na] && isfreenode[nb]
                # determine upwind direction
                up = (P[j2] - P[j1]) >= 0
                # sign = +1 if Vn[i]>0, else -1
                sgn = Vn[i] > 0 ? 1.0 : -1.0
                d_na = sgn * ( up ?  f[nb]  :  -f[na] )
                ∂Vn[i] += d_na * ΔR[j1]
                up₂ = (P[j1] - P[j2]) >= 0
                d_nb = sgn * ( up₂ ?  f[na]  :  -f[nb] )
                ∂Vn[i] += d_nb * ΔR[j2]
            end
        end
        return ChainRulesCore.NoTangent(),
               @ChainRulesCore.thunk(∂f),       # ∂L/∂f
               @ChainRulesCore.thunk(∂Qs),      # ∂L/∂Qs
               ChainRulesCore.NoTangent(),      # ∂L/∂neighbors
               @ChainRulesCore.thunk(∂P),       # ∂L/∂P  (zero here)
               @ChainRulesCore.thunk(∂Vn)       # ∂L/∂Vn
    end
    return R, pullback
end

#function to implement two point flux approximation. Calculates pressure and darcy velocity
function tpfa(Ks,dirichleths, dirichletnodes, Qs,areasoverlengths,neighbors)
    Ks2Ks_neighbors(Ks) = ( 0.5*(Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
    Ks_neighbors = Ks2Ks_neighbors(Ks)
    P=solvepressure(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
    tx = 2*areasoverlengths; 
    P_diff_neighbors(P) = ((P[map(p->p[1], neighbors)] .- P[map(p->p[2], neighbors)]))
    P_n = P_diff_neighbors(P)
    Vn = [P_n[i] * (tx[i] * Ks_neighbors[i]) for i in 1:length(neighbors)]
    return P,  Vn
end

#function to calculate total mobility, which accounts the relative permeability
function relativeperm(s,fluid)
    S = (s.-fluid.swc)/(1-fluid.swc-fluid.sor); Mw = S.^2/fluid.vw;
    Mo =(1 .- S).^2/fluid.vo;
    return Mw, Mo
end

#function to calculate saturation using the pressure. Convergence is assured by CFL condition
function upstream( S, fluid,  Qs, T, P,Vn,neighbors,volumes)
    porosity = ones(size(volumes))
    pv = volumes .* porosity[:];
    fi = max.(Qs, 0)
    # Compute the minimum pore volume / velocity ratio for all cells
    Vi = zeros(length(pv))  # Total velocity (flux) for each cell
    for (i, (node_a, node_b)) in enumerate(neighbors) 
        if Vn[i]<0
            Vi-=Float64.(([cval == node_a for cval in 1:length(Vi)]).*(Vn[i]))
        else
            Vi+=Float64.(([cval == node_b for cval in 1:length(Vi)]).*(Vn[i]))
        end
    end
    pm = minimum(pv ./ (Vi + fi)) # 1e-8 is for handling NAN
    # CFL time step based on saturation upstreaming
    cfl = ((1 - fluid.swc - fluid.sor) / 3) * pm
    Nts = ceil(Int, T/cfl) 
    dtx = (T / Nts) ./ pv  # Time step for each cell
    for i=1:Nts
        mw, mo = relativeperm(S, fluid)
        f = mw ./ (mw + mo)
        fi = max.(Qs,0).*dtx  
        S+= saturationcalc_residuals(f,Qs,neighbors,P,Vn) .* dtx + fi ;
        # enforce physical bounds
        S = clamp.(S, fluid.swc, 1 - fluid.sor)
    end
    return S
end

# #time series solver
function solvetwophase(args...)
    h0, S0, K,dirichleths,  dirichletnodes, Qs,  volumes, areasoverlengths, fluid, dt, neighbors, nt, everyStep =args
    if everyStep
        P_data = []
        S_data = []
    end
    S = S0
    P = h0
    for t =1:nt
        Mw, Mo = relativeperm(S, fluid)
        Mt = Mw .+ Mo 
        Km=Mt.*K
        P, Vn = tpfa(Km,dirichleths, dirichletnodes, Qs, areasoverlengths,neighbors)
        S = upstream(S, fluid, Qs, dt, P, Vn, neighbors, volumes)
        if everyStep
            @show t,sum(S),sum(P)
            push!(P_data, deepcopy(P))
            push!(S_data, deepcopy(S))
        end
    end
    if everyStep
        return P_data, S_data
    else
        return P, S #Return the results from the last 
    end
end

