import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#=============================================================================
# Initialize attributes and define the Lorenz system
#=============================================================================

# The Lorenz system is a 3D system of autonomous differential equations of the
# form:
#
#     x' = sigma (y - x)
#     y' = x (rho - z) - y
#     z' = x y - beta z

# Problem parameters

sigma = 10 # sigma parameter
rho = 28 # rho parameter
beta = 8/3 # beta parameter
x0 = 1 # initial x value
y0 = 1 # initial y value
z0 = 1 # initial z value
ti = 0 # initial time value
tf = 50 # final time value

# The numerical ODE solver we'll be using below requires a function which
# defines the righthand side of the differential equation, i.e. the function F
# in the ODE x' = F(t,x).
# This can be provided either by defining a local function and supplying its
# handle, or by defining it as a lambda function within the solver's arguments.
# We will be defining it a function "rhs" below.

# Define a function to represent the righthand side of the Lorenz system
def rhs(T, X):
    
    dx = np.array([0, 0, 0]) # initialize output array
    dx[0] = sigma*(X[1] - X[0])
    dx[1] = X[0]*(rho - X[2]) - X[1]
    dx[2] = X[0]*X[1] - beta*X[2]
    return dx

# Note that the above could be accomplished in a single line as follows:
# rhs = lambda T, X: np.array([sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2);
#                              X(1)*X(2) - beta*X(3)])

#=============================================================================
# Numerically solve the Lorenz system
#=============================================================================

# The Lorenz system is numerically solved using SciPy's "solve_ivp" function,
# a general initial value problem-solving API that includes several methods.
# We will be using its implementation of RK45, an explicit Runge-Kutta (4,5)
# method.
# SciPy also includes a function, "RK45", that is roughly equivalent to
# MATLAB's "ode45" function, but solve_ivp is preferred because it is more
# general and includes many other methods.

# The SciPy numerical ODE solver's options are passed as individual keyword
# arguments, and need not be defined here.

# The arguments of solve_ivp include, respectively: the function of time T and
# the vector of state variables X which defines the righthand side of the ODE
# system, followed by a time interval as a tuple (initial, final), followed
# by an array-like list of initial values [x0, y0, z0]. Many additional
# optional arguments exist, including a keyword argument to specify the
# method used by the solver, as well as a variety of keyword arguments to tune
# the behavior of the solver.

sol = solve_ivp(rhs, (ti, tf), [x0, y0, z0], method="RK45", rtol=0.000001)
T = sol.t # get time attribute from solution object
X = sol.y # get state variable attribute from solution object

# T is now an array of time values, while X is an array whose columns contain
# the numerical values of the x, y, and z variables.

# Note that, if we had not defined the function rhs above, we could include
# it as a lambda function here by replacing the first argument with:
# lambda T, X: np.array([sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2);
#                        X(1)*X(2) - beta*X(3)])
# This may sometimes be worthwhile if the ODE in question is very simple.

#=============================================================================
# Display various projections of the solution curve
#=============================================================================

###
# Note that, due to the chaotic nature of the Lorenz system, the numerical
# solutions from MATLAB and Python are expected to be slightly different.
print(T)
print(X[0])
print(X[1])
print(X[2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[0], X[1], X[2])
plt.show()
input("...")
