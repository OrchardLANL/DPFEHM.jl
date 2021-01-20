using Flux
using Statistics: mean, std
using Random
#import MultiTheisDP.MultiTheis
#using PyPlot
#ioff()
#using PyCall
#gridspec = pyimport("matplotlib.gridspec")
#using LaTeXStrings
using JLD2
#include("plot_train.jl")
#include("theislike.jl")
push!(LOAD_PATH,"../../src")
push!(LOAD_PATH,"./")
import TheisLike

########################################################
# User input
# Location to monitor pressure at
mon_well_crds = [[-75.0,-75.0]]
# Location of injection and extraction wells
well_crds = [[-50.0,-50.0],
			[50.0,50.0]]
# Injection rate
Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
# Time to check pressure
t = 30.0*86400.0 # 30 days converted to seconds
# Create xy grid for plotting pressures
xs = ys = -100.0:1.0:100.0
# Training epochs
epochs = 1:120
########################################################

# Calculate distance between extraction and injection wells
rs = [sqrt((sum((cs-mon_well_crds[1]).^2))) for cs in well_crds]

# Input/output normalization function
lerp(x, lo, hi) = x*(hi-lo)+lo

global nevals = 0
# Wrapper around MultiTheis solve returning pressure head at a point
function head(T::Float64,S::Float64,Q1::Float64,Qinj::Float64,rs::Array{Float64,1},t::Float64;train=false)
	Qs = [Q1,Qinj]
	if train
		global nevals = nevals + 1
	end
	#MultiTheis.solve(Qs,T,S,t,rs)
	TheisLike.solve_numerical(Qs, T, S, t, rs)
end

# This helper function doesn't work with DPFEHM. Not really needed anyhow.
#function head(T::Number,S::Number,Q1::Number;train=true)
#	# Injection rate
#	Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
#	# Time to check pressure
#	t = 30.0*86400.0 # 30 days converted to seconds
#	# Calculate distance between extraction and injection wells
#	rs = [sqrt((sum((cs-mon_well_crds[1]).^2))) for cs in well_crds]
#	#Qs = [Q1,Qinj]
#	return head(T,S,Q1,Qinj,rs,t,train=train)
#end

# Set random seed for repeatability
Random.seed!(0)

# Neural network structure
model = Chain(Dense(3, 8, σ),
              Dense(8, 16, σ),
              Dense(16, 8, σ),
              Dense(8, 1)) |> f64

# Make neural network parameters trackable by Flux
θ = params(model)

# Neural network function (T, S, target)->(Q1,Q2)
# 'target' is the target pressure head at the monitoring location
function aim(T, S, target)
  Q1 = model([T, S, target])
  # Make is so that Q1 and Q2 are always negative for extraction
  Q1 = -softplus(Q1[1])
  #Q2 = -softplus(Q2)
  Q1
end

# Set ranges for P, T, S
# Fix pressure target for now. It should be removed, but when I removed it with MultiTheis, the PIML framework didn't work as well. Moral of the story, NN's are squirrelly things that don't always make sense. 
PRES = (0.0,0.0) # Pressure target range [m] 
TRANS = (-2.0,0.0) # log10 transmissivity [m^2/s]
# Assume that reservoir thickness is 100m (S = S_s*b)
STOR = (-4.0,-1.0) # log10 storativity

# Random T, S, and P generator
sample() = [10^lerp(rand(), TRANS...), 10^lerp(rand(), STOR...), lerp(rand(), PRES...)]

# Physics model vs NN 
function mydiff(T, S, target; train=false)
    (head(T, S, aim(T, S, target), Qinj, rs, t, train=train) - target)
end
function loss(T, S, target; train=true)
	mydiff(T, S, target, train=train)^2
end
loss(x; train=true) = sum(map(x->loss(x[1],x[2],x[3];train=train),x))
mydiff(x; train=true) = sum(map(x->mydiff(x[1],x[2],x[3];train=train),x))

function cb2()
	# Terminal output
	loss_train = sum(map(x->loss(x,train=false),data_train_batch))
	loss_test = sum(map(x->loss(x,train=false),data_test))
	println(string("callback: train loss: ", loss_train, " test loss: ", loss_test))
end

global losses_train = Float64[]
global losses_test = Float64[]
global rmses_train = Float64[]
global rmses_test = Float64[]
global meandiffs_train = Float64[]
global stddiffs_train = Float64[]
global meandiffs_test = Float64[]
global stddiffs_test = Float64[]
global train_time = Float64[]

cb = Flux.throttle(cb2, 10)

opt = ADAM()
