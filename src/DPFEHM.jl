module DPFEHM

import AlgebraicMultigrid
import Calculus
import ChainRulesCore
import ForwardDiff
import Interpolations
import NLsolve
import NonlinearEquations
import SparseArrays
import StaticArrays

include("grid.jl")
include("groundwater.jl")
include("RelPerm.jl")
include("richards.jl")

end
