#this performs inverse analysis repeatedly
#the inverse problem is parameterized by coefficients of the eigenvectors of the permeability field
#in the end, it produces a plot showing the distribution of how well the inverse analysis performs for each of the eigenvector coefficients
#the example is somewhat computationally intensive
#it is recommended to run this example in parallel by starting julia with "julia -p 32" where 32 processors will be used
include("weeds.jl")

#run the inverse analysis repeatedly for a set of parameters
sigma, lambda, mu, deltah, sqrtnumobs, alpha, beta, gamma, iterations = 1.0, 50.0, 0.0, 5.0, 16, 1.0, 1.0, 1.0, 100#here are the parameters we will run the inverse analysis with
@time results = pmap(x->inversemodel(sigma, lambda, mu, deltah, sqrtnumobs, alpha, beta, gamma, iterations), 1:100)#run the inverse analysis 100 times (in parallel if julia is run like, e.g., "julia -p 8" where 8 processors will be used)

#postprocess the results
eig_residuals = map(x->x[1], results)
eig_rmse = sum(map(x->x .^ 2, eig_residuals)) / length(eig_residuals)
logK_rmse = map(x->x[2], results)
of_vals = map(x->x[3], results)

#make a couple plots based on the results
fig, ax = PyPlot.subplots()
ax.semilogy(eig_rmse, ".")
ax.set_xlabel("Eigenvector Index")
ax.set_ylabel("Eigenvector Coefficient RMSE")
PyPlot.savefig("eigenvector_coefficient_fit.pdf")
display(fig)
println()
PyPlot.close(fig)

fig, ax = PyPlot.subplots()
for i = 1:length(logK_rmse)
	ax.loglog(of_vals[i], logK_rmse[i], ".", alpha=0.1)
end
ax.set_xlabel("Objective Function")
ax.set_ylabel("L2 Error in logK")
PyPlot.savefig("logK_vs_objective_function.pdf")
display(fig)
println()
PyPlot.close(fig)
