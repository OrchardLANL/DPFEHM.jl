import AlgebraicMultigrid
import DPFEHM
import FEHM
import HDF5
import ILUZero
import IterativeSolvers
import LinearAlgebra
import Preconditioners

import Base.eltype
import Base.*

function loaddir(dirname)
	coords, volumes, neighbors, areas, lengths = DPFEHM.load_uge("$dirname/full_mesh_vol_area.uge")
	areasoverlengths = areas ./ lengths
	leftnodes = FEHM.readzone("$dirname/pboundary_left_w.zone")[2][1]
	rightnodes = FEHM.readzone("$dirname/pboundary_right_e.zone")[2][1]
	dirichletnodes = [leftnodes; rightnodes]
	dirichleths = [2e6 * ones(length(leftnodes)); 1e6 * ones(length(rightnodes))]
	Qs = zeros(size(coords, 2))
	Ks = HDF5.h5read("$dirname/dfn_properties.h5", "Permeability")
	Ks2Ks_neighbors(Ks) = sqrt.((Ks[map(p->p[1], neighbors)] .* Ks[map(p->p[2], neighbors)]))#convert from permeabilities at the nodes to permeabilities connecting the nodes
	Ks_neighbors = Ks2Ks_neighbors(Ks)
	dirichleths2 = zeros(length(Qs))
	for (i, j) in enumerate(dirichletnodes)
		dirichleths2[j] = dirichleths[i]
	end
	args = (zeros(length(Qs) - length(dirichletnodes)), Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths2, Qs, ones(length(Qs)), ones(length(Qs)))
	b = -DPFEHM.groundwater_residuals(args...)
	A = DPFEHM.groundwater_h(args...)
	return A, b, (neighbors, areasoverlengths, Ks_neighbors, Qs, dirichletnodes, dirichleths)
end

preconditioners = [("AMGSA", A->AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(A))),
				   ("ILU", A->ILUZero.ilu0(A)),
				   ("Iden", A->LinearAlgebra.I)]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
markers = ["", "--", "+", ":", "."]
preconditioned_solvers = [
	("CG", (A, b, pl)->begin
		x, ch = IterativeSolvers.cg(A, b; maxiter=10000, reltol=1e-12, log=true, Pl=pl)
		return x, ch.data[:resnorm]
	end),
	("GMRES", (A, b, pl)->begin
		x, ch = IterativeSolvers.gmres(A, b; maxiter=10000, reltol=1e-12, log=true, Pl=pl)
		return x, ch.data[:resnorm]
	end),
	("BiCGStab", (A, b, pl)->begin
		x, ch = IterativeSolvers.bicgstabl(A, b; max_mv_products=10000, reltol=1e-12, log=true, Pl=pl)
		return x, ch.data[:resnorm]
	end)]
unpreconditioned_solvers = [
	("AMGSA", (A, b)->begin
		ml = AlgebraicMultigrid.smoothed_aggregation(A)
		x, ch = AlgebraicMultigrid.solve(ml, b; maxiter=10000, reltol=1e-12, log=true)
		return x, ch
	end),
	("MINRES", (A, b)->begin
		x, ch = IterativeSolvers.minres(A, b; maxiter=10000, reltol=1e-12, log=true)
		return x, ch.data[:resnorm]
	end),
	("QMR", (A, b)->begin
		x, ch = IterativeSolvers.qmr(A, b; maxiter=10000, reltol=1e-12, log=true)
		return x, ch.data[:resnorm]
	end),
	("Cholesky", (A, b)->begin
		Af = LinearAlgebra.cholesky(A)
		x = Af \ b
		return x, [1.0, sqrt(sum((A * x .- b) .^ 2) / sum(b .^ 2))]
   end)]
