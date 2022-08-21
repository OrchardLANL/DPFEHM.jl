using Test
@testset "Relative Permeability" begin
	include("corey.jl")
end
@testset "Grids" begin
	include("grid.jl")
end
@testset "Theis solutions" begin
	include("theis.jl")
end
@testset "Thiem solutions" begin
	include("thiem.jl")
end
@testset "Transport equation" begin
	include("transport.jl")
end
@testset "Wave equation" begin
	include("wave.jl")
end
