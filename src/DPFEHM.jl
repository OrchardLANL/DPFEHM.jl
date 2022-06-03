module DPFEHM

import AlgebraicMultigrid
import ChainRulesCore
import DelimitedFiles
import ForwardDiff
import Interpolations
import LinearAlgebra
import NLsolve
import NonlinearEquations
import SparseArrays
import StaticArrays

include("grid.jl")
include("groundwater.jl")
include("RelPerm.jl")
include("richards.jl")
include("transport.jl")
include("wave.jl")
include("utilities.jl")

end
