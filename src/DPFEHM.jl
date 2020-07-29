module DPFEHM

import Calculus
import ForwardDiff
import NonlinearEquations
import SparseArrays
import StaticArrays

include("grid.jl")
include("groundwater.jl")
include("richards.jl")

end
