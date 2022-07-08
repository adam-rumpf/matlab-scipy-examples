import numpy as np                                                               # 
import matplotlib.pyplot as plt                                                  # 
from scipy.integrate import solve_ivp                                            # 
                                                                                 # 
#=============================================================================   # %=============================================================================
# Initialize attributes and define the Lorenz system                             # % Initialize attributes and define the Lorenz system
#=============================================================================   # %=============================================================================
                                                                                 # 
# The Lorenz system is a 3D system of autonomous differential equations of the   # % The Lorenz system is a 3D system of autonomous differential equations of the
# form:                                                                          # % form:
#                                                                                # %
#     x' = sigma (y - x)                                                         # %     x' = sigma (y - x)
#     y' = x (rho - z) - y                                                       # %     y' = x (rho - z) - y
#     z' = x y - beta z                                                          # %     z' = x y - beta z
                                                                                 # 
# Problem parameters                                                             # % Problem parameters
                                                                                 # global sigma rho beta; % declare parameters as global for use in the function
sigma = 10 # sigma parameter                                                     # sigma = 10; % sigma parameter
rho = 28 # rho parameter                                                         # rho = 28; % rho parameter
beta = 8/3 # beta parameter                                                      # beta = 8/3; % beta parameter
x0 = 1 # initial x value                                                         # x0 = 1; % initial x value
y0 = 1 # initial y value                                                         # y0 = 1; % initial y value
z0 = 1 # initial z value                                                         # z0 = 1; % initial z value
ti = 0 # initial time value                                                      # ti = 0; % initial time value
tf = 50 # final time value                                                       # tf = 50; % final time value
                                                                                 # 
# The numerical ODE solver we'll be using below requires a function which        # % The numerical ODE solver we'll be using below requires a function which
# defines the righthand side of the differential equation, i.e. the function F   # % defines the righthand side of the differential equation, i.e. the function F
# in the ODE x' = F(t,x).                                                        # % in the ODE x' = F(t,x).
# This can be provided either by defining a local function and supplying its     # % This can be provided either by defining a local function and supplying its
# handle, or by defining it as a lambda function within the solver's arguments.  # % handle, or by defining it as an inline function within the solver's arguments.
# We will be defining it a function "rhs" below.                                 # % We will be defining it a local function "rhs" below.
                                                                                 # 
# Define a function to represent the righthand side of the Lorenz system         # % Define a function to represent the righthand side of the Lorenz system
def rhs(T, X):                                                                   # function dx = rhs(T, X)
                                                                                 #     global sigma rho beta; % declare global variables
    dx = np.array([0, 0, 0]) # initialize output array                           #     dx = [0; 0; 0]; % initialize output list
    dx[0] = sigma*(X[1] - X[0])                                                  #     dx(1) = sigma*(X(2) - X(1));
    dx[1] = X[0]*(rho - X[2]) - X[1]                                             #     dx(2) = X(1)*(rho - X(3)) - X(2);
    dx[2] = X[0]*X[1] - beta*X[2]                                                #     dx(3) = X(1)*X(2) - beta*X(3);
    return dx                                                                    # end
                                                                                 # 
# Note that the above could be accomplished in a single line as follows:         # % Note that the above could be accomplished in a single line as follows:
# rhs = lambda T, X: np.array([sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2);    # % rhs = @(T,X) [sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2); ...
#                              X(1)*X(2) - beta*X(3)])                           # %               X(1)*X(2) - beta*X(3)];
                                                                                 # 
#=============================================================================   # %=============================================================================
# Numerically solve the Lorenz system                                            # % Numerically solve the Lorenz system
#=============================================================================   # %=============================================================================
                                                                                 # 
# The Lorenz system is numerically solved using SciPy's "solve_ivp" function,    # % The Lorenz system is numerically solved using MATLAB's "ode45" function, a
# a general initial value problem-solving API that includes several methods.     # % widely-used implementation of an explicit Runge-Kutta (4,5) method.
# We will be using its implementation of RK45, an explicit Runge-Kutta (4,5)     # % MATLAB has no unified API for numerically solving ODEs, and instead each
# method.                                                                        # % different numerical method is implemented as a separate function with some
# SciPy also includes a function, "RK45", that is roughly equivalent to          # % tuneable parameters.
# MATLAB's "ode45" function, but solve_ivp is preferred because it is more       # 
# general and includes many other methods.                                       # 
                                                                                 # 
# The SciPy numerical ODE solver's options are passed as individual keyword      # % Set precision for ODE solver
# arguments, and need not be defined here.                                       # options = odeset("RelTol", 0.0000001);
                                                                                 # 
# The arguments of solve_ivp include, respectively: the function of time T and   # % The arguments of ode45 include, respectively: the function of time T and
# the vector of state variables X which defines the righthand side of the ODE    # % the vector of state variables X which defines the righthand side of the
# system, followed by a time interval as a tuple (initial, final), followed      # % ODE system, followed by a time interval of the form [initial, final],
# by an array-like list of initial values [x0, y0, z0]. Many additional          # % followed by a list of initial values of the form [x0, y0, z0]. The optional
# optional arguments exist, including a keyword argument to specify the          # % fourth argument is a set of ODE solver options.
# method used by the solver, as well as a variety of keyword arguments to tune   # 
# the behavior of the solver.                                                    # 
                                                                                 # 
sol = solve_ivp(rhs, (ti, tf), [x0, y0, z0], method="RK45", rtol=0.000001)       # [T, X] = ode45(@(T,X) rhs(T, X), [ti, tf], [x0, y0, z0], options);
T = sol.t # get time attribute from solution object                              # 
X = sol.y # get state variable attribute from solution object                    # 
                                                                                 # 
# T is now an array of time values, while X is an array whose columns contain    # % T is now a list of time values, while X is a matrix whose columns contain
# the numerical values of the x, y, and z variables.                             # % the numerical values of the x, y, and z variables.
                                                                                 # 
# Note that, if we had not defined the function rhs above, we could include      # % Note that, if we had not defined the function rhs above, we could include
# it as a lambda function here by replacing the first argument with:             # % it as an inline function here by replacing the first argument with:
# lambda T, X: np.array([sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2);          # % @(T,X) [sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2); ...
#                        X(1)*X(2) - beta*X(3)])                                 # %         X(1)*X(2) - beta*X(3)]
# This may sometimes be worthwhile if the ODE in question is very simple.        # % This may sometimes be worthwhile if the ODE in question is very simple.
                                                                                 # 
#=============================================================================   # %=============================================================================
# Display various projections of the solution curve                              # % Display various projections of the solution curve
#=============================================================================   # %=============================================================================
                                                                                 # 
# Note that, due to the chaotic nature of the Lorenz system, the numerical       # % Note that, due to the chaotic nature of the Lorenz system, the numerical
# solutions from MATLAB and Python are expected to be slightly different.        # % solutions from MATLAB and Python are expected to be slightly different.
                                                                                 # 
fig = plt.figure() # create a new figure object and store handle                 # figure % create a new figure window
                                                                                 # 
# Show xy-projection                                                             # % Show xy-projection
plt.subplot(2, 2, 1) # target first (NW) tile in a 2x2 layout                    # subplot(2, 2, 1) % target first (NW) tile in a 2x2 layout
plt.plot(X[:][0], X[:][1], linewidth=0.5) # plot y versus x                      # plot(X(:,1), X(:,2), "LineWidth", 0.5) % plot y versus x
plt.xlabel("x")                                                                  # xlabel("x")
plt.ylabel("y")                                                                  # ylabel("y")
                                                                                 # 
# Show xz-projection                                                             # % Show xz-projection
plt.subplot(2, 2, 3) # target third (SW) tile in a 2x2 layout                    # subplot(2, 2, 3) % target third (SW) tile in a 2x2 layout
plt.plot(X[:][0], X[:][2], linewidth=0.5) # plot z versus x                      # plot(X(:,1), X(:,3), "LineWidth", 0.5) % plot z versus x
plt.xlabel("x")                                                                  # xlabel("x")
plt.ylabel("z")                                                                  # ylabel("z")
                                                                                 # 
# Show yz-projection                                                             # % Show yz-projection
plt.subplot(2, 2, 2) # target second (NE) tile in a 2x2 layout                   # subplot(2, 2, 2) % target second (NE) tile in a 2x2 layout
plt.plot(X[:][1], X[:][2], linewidth=0.5) # plot z versus y                      # plot(X(:,2), X(:,3), "LineWidth", 0.5) % plot z versus y
plt.xlabel("y")                                                                  # xlabel("y")
plt.ylabel("z")                                                                  # ylabel("z")
                                                                                 # 
# Set up axes for oblique projection                                             # % Set up axes for oblique projection
ax = fig.add_subplot(2, 2, 4, projection="3d") # 3D axes in fourth (SE) tile     # subplot(2, 2, 4) % target fourth (SE) tile in a 2x2 layout
ax.xaxis.pane.fill = False # remove fill colors from axis planes                 # 
ax.yaxis.pane.fill = False                                                       # % In the Python version of this script, additional lines are required to
ax.zaxis.pane.fill = False                                                       # % remove the default gray planes and grid lines included in 3D plots.
ax.xaxis.pane.set_edgecolor("w") # set axis plane edge colors to white           # % MATLAB does not automatically include these.
ax.yaxis.pane.set_edgecolor("w")                                                 # 
ax.zaxis.pane.set_edgecolor("w")                                                 # 
ax.grid(False) # remove grid lines from axis planes                              # 
                                                                                 # 
# Generate 3D plot for oblique projection                                        # % Generate 3D plot for oblique projection
ax.plot3D(X[:][0], X[:][1], X[:][2], linewidth=1) # plot coordinates on 3D axes  # plot3(X(:,1), X(:,2), X(:,3), "LineWidth", 1) % plot 3D coordinates
ax.set_xlabel("x")                                                               # xlabel("x")
ax.set_ylabel("y")                                                               # ylabel("y")
ax.set_zlabel("z")                                                               # zlabel("z")
ax.view_init(30, 135) # set view angle elevation and azimuth (respectively)      # view([225, 30]) % set view angle azimuth and elevation (respectively)
                                                                                 # 
# Add a title to the window                                                      # % Add a title to an empty window
fig.suptitle("The Lorenz Attractor")                                             # axes("visible", "off", "title", "The Lorenz Attractor")
                                                                                 # 
# Display the figure window                                                      # % MATLAB will automatically display the resulting figure.
fig.tight_layout() # automatically adjust subfigure margins to fit all text      # 
fig.show() # show the figure                                                     # % Note that, as of MATLAB R2018b, a subplot grid can be more easily titled by
                                                                                 # % using the "sgtitle" function. This approach will work for older versions,
                                                                                 # % and also for Octave.