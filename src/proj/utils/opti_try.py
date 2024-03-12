# Import packages.
import cvxpy as cp
import numpy as np

# Generate data.
m = 2
n = 2
np.random.seed(1)
A = np.eye(m)


vals, vecs = np.linalg.eig(A)

print(vals)
# # Define and solve the CVXPY problem.
# x = cp.Variable(n)
# cost = -cp.sum_squares(A@x)
# prob = cp.Problem(cp.Minimize(cost))
# prob.solve()

# # Print result.
# print("\nThe optimal value is", prob.value)
# print("The optimal x is")
# print(x.value)
# print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)