import Random
Random.seed!(5)
include("weeds_fracture_3D.jl")
x_test = randn(3 * num_eigenvectors + 2) #the input to the function should follow an N(0, I) distribution -- the eigenvectors and eigenvalues are embedded inside f
@time @show f(x_test) #compute the pressure at the critical location
@time @show extrema(grad_f(x_test)) #compute the gradient of the pressure at the critical location with respect to the eigenvector coefficients (then only show the largest/smallest values so it doesn't print a huge array)

#Quick demo of how you could do Monte Carlo to get the distribution of pressures at the critical location -- hopefully the method Georg described can help us understand the tail distribution better
numruns = 10 ^ 2
fs = Float64[]
@time for i = 1:numruns
    x_test = randn(3 * num_eigenvectors + 2)
    @time push!(fs, f(x_test))
    @show i
end
@show sum(fs) / length(fs)
@show extrema(fs)

fig, axs = PyPlot.subplots(1, 2, figsize=(16, 9))
axs[1].hist(fs, bins=30)
axs[1].set(xlabel="Pressure [MPa] at critical location", ylabel="Count (out of $(numruns))")
axs[2].hist(log.(fs), bins=30)
axs[2].set(xlabel="log(Pressure [MPa]) at critical location", ylabel="Count (out of $(numruns))")
fig.tight_layout()
display(fig); println()
fig.show()
fig.savefig("distribution_of_pressure.png")
PyPlot.close(fig)
