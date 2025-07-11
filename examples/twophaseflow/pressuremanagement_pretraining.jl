#HR edit
#This example trains a neural network to manage the pressure in an underground reservoir.
#The ML model gets to control the pumping rate at an extraction well.
#It's goal is to manage the pressure at a critical location so the pressure increase caused by a fixed injection at another well is mitigated by the extraction.
#It uses a subsurface flow model in the loop to train a convolutional neural network to predict the extraction rate given a permeability field.
import DPFEHM
import GaussianRandomFields
import Optim
import Random
import BSON
import Zygote
import ChainRulesCore
import Flux


global losses_train_pt = Float64[]
global losses_test_pt = Float64[]
global rmses_train_pt = Float64[]
global rmses_test_pt = Float64[]
global train_time_pt = Float64[]


# Injection rate
Qinj = 0.031688 # [m^3/s] (1 MMT water/yr)
n = 26
ns = [n, n]
steadyhead = 0e0
sidelength = 100
thickness  = 1.0
mins = [-sidelength, -sidelength] #meters
maxs = [sidelength, sidelength] #meters
num_eigenvectors = 200
sigma = 1.0
lambda = 50
cov = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda, 1; σ=sigma))
x_pts = range(mins[1], maxs[1]; length=ns[1])
y_pts = range(mins[2], maxs[2]; length=ns[2])
grf = GaussianRandomFields.GaussianRandomField(cov, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), x_pts, y_pts)

pressure_target  = 0
learning_rate = 1e-4

coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, thickness)
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
  
# Calculate distance between extraction and injection wells
monitoring_well_node = 190
@assert coords[:, monitoring_well_node] == [-44, -44]
injection_extraction_nodes = [271, 487]
@assert coords[:, injection_extraction_nodes[1]] == [-20, -20]
@assert coords[:, injection_extraction_nodes[2]] == [44, 44]


for i = 1:size(coords, 2)
    if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

function getQs(Qs::Vector, is::Vector)
    sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
end

function solve_numerical(Qs, T)
    logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
    Qs = getQs(Qs, injection_extraction_nodes)
    Ks_neighbors = logKs2Ks_neighbors(T)
    h_gw = DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
    return h_gw[monitoring_well_node]- steadyhead
end


#like LeNet5
model = Flux.Chain(Flux.Conv((5, 5), 1=>6, Flux.relu),
              Flux.MaxPool((2, 2)),
              Flux.Conv((5, 5), 6=>16, Flux.relu),
              Flux.MaxPool((2, 2)),
              Flux.flatten,
              Flux.Dense(144, 120, Flux.relu),
              Flux.Dense(120, 84, Flux.relu),
              Flux.Dense(84, 1)) |> Flux.f64

# Make neural network parameters trackable by Flux
θ = Flux.params(model)

function loss(x)
    Ts = reshape(hcat(map(y->y[1], x)...), size(x[1][1], 1), size(x[1][1], 2), 1, length(x))
    targets = map(y->y[2], x)
    Q1 = model(Ts)
    Qs = map(Q->[Q, Qinj], Q1)
    loss = sum(map(i->solve_numerical(Qs[i], Ts[:, :, 1, i]) - targets[i], 1:size(Ts, 4)).^2)
    return loss
end

opt = Flux.ADAM(learning_rate)

# Training epochs
epochs = 1:4000
batch_size = 15



data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for v in 1:300/batch_size]
data_test = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for i = 1:300/batch_size]
println("The training has started..")
loss_train = sum(map(x->loss(x), data_train_batch))
rmse_train = sqrt(loss_train/(batch_size *length(data_train_batch)))
loss_test = sum(map(x->loss(x), data_test))
rmse_test = sqrt(loss_test/(batch_size *length(data_test)))
println(string("epoch: 0 train rmse: ", rmse_train, " test rmse: ", rmse_test))
# Save convergence metrics
push!(losses_test_pt, loss_test)
push!(rmses_test_pt, rmse_test)

for epoch in epochs
    data_train_batch = [[(GaussianRandomFields.sample(grf), pressure_target) for i = 1:batch_size] for v in 1:300/batch_size]
    tt = @elapsed Flux.train!(loss, θ, data_train_batch, opt)
    push!(train_time_pt, tt)
    loss_train = sum(map(x->loss(x), data_train_batch))
    rmse_train = sqrt(loss_train/(batch_size *length(data_train_batch)))
    loss_test = sum(map(x->loss(x), data_test))
    rmse_test = sqrt(loss_test/(batch_size *length(data_test)))

    # Terminal output
    println(string("epoch: ", epoch, " time: ", tt, " train rmse: ", rmse_train, " test rmse: ", rmse_test))
    push!(losses_train_pt, loss_train)
    push!(rmses_train_pt, rmse_train)
    push!(losses_test_pt, loss_test)
    push!(rmses_test_pt, rmse_test)
end
@BSON.save "model_preTraining_$(batch_size)_$(learning_rate).bson"  epochs train_time_pt losses_train_pt  rmses_train_pt  losses_test_pt  rmses_test_pt  
@BSON.save "mytrained_model.bson" model

