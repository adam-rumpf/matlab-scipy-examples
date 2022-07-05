


%=============================================================================
% Initialize attributes and define the Lorenz system
%=============================================================================

% The Lorenz system is a 3D system of autonomous differential equations of the
% form:
%
%     x' = sigma (y - x)
%     y' = x (rho - z) - y
%     z' = x y - beta z

% Problem parameters
global sigma rho beta; % declare parameters as global for use in the function
sigma = 10; % sigma parameter
rho = 28; % rho parameter
beta = 8/3; % beta parameter
x0 = 1; % initial x value
y0 = 1; % initial y value
z0 = 1; % initial z value
ti = 0; % initial time value
tf = 50; % final time value

% Define a function to represent the righthand side of the Lorenz system
function dx = rhs(T, X)
	global sigma rho beta; % declare global variables
	dx = [0; 0; 0]; % initialize output list
	dx(1) = sigma*(X(2) - X(1));
	dx(2) = X(1)*(rho - X(3)) - X(2);
	dx(3) = X(1)*X(2) - beta*X(3);
end

%=============================================================================
% Numerically solve the Lorenz system
%=============================================================================

% The Lorenz system is numerically solved using MATLAB's "ode45" function, a
% widely-used implementation of an explicit Runge-Kutta (4,5) method.

% Set precision for ODE solver
options = odeset("RelTol", 0.0000001);

% The first argument of ode45 is a function handle or an inline function. The
% first argument of this function must always be time, while the second must
% always be a vector of state variables. In MATLAB this can be accomplished
% either by defining a local function (as we've done above with the "rhs"
% function) or by defining an inline function within ode45, itself.
% The second argument is a list of time values of the form [initial, final].
% The third is a list of initial values of the form [x0, y0, z0].
% The fourth is a set of ODE solver options.

[T, X] = ode45(@(T,X) rhs(T, X), [ti, tf], [x0, y0, z0], options);

%=============================================================================
% Display various projections of the solution curve
%=============================================================================
