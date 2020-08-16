module DPFEHM

import Calculus
import ForwardDiff
import NonlinearEquations
import SparseArrays
import StaticArrays

include("grid.jl")
include("groundwater.jl")
include("RelPerm.jl")
include("richards.jl")

end
