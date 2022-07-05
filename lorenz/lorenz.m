



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

% The numerical ODE solver we'll be using below requires a function which
% defines the righthand side of the differential equation, i.e. the function F
% in the ODE x' = F(t,x).
% This can be provided either by defining a local function and supplying its
% handle, or by defining it as an inline function within the solver's arguments.
% We will be defining it a local function "rhs" below.

% Define a function to represent the righthand side of the Lorenz system
function dx = rhs(T, X)
	global sigma rho beta; % declare global variables
	dx = [0; 0; 0]; % initialize output list
	dx(1) = sigma*(X(2) - X(1));
	dx(2) = X(1)*(rho - X(3)) - X(2);
	dx(3) = X(1)*X(2) - beta*X(3);
end

% Note that the above could be accomplished in a single line as follows:
% rhs = @(T,X) [sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2); ...
%               X(1)*X(2) - beta*X(3)];

%=============================================================================
% Numerically solve the Lorenz system
%=============================================================================

% The Lorenz system is numerically solved using MATLAB's "ode45" function, a
% widely-used implementation of an explicit Runge-Kutta (4,5) method.
% MATLAB has no unified API for numerically solving ODEs, and instead each
% different numerical method is implemented as a separate function with some
% tuneable parameters.



% Set precision for ODE solver
options = odeset("RelTol", 0.0000001);

% The arguments of ode45 include, respectively: the function of time T and
% the vector of state variables X which defines the righthand side of the
% ODE system, followed by a time interval of the form [initial, final],
% followed by a list of initial values of the form [x0, y0, z0]. The optional
% fourth argument is a set of ODE solver options.



[T, X] = ode45(@(T,X) rhs(T, X), [ti, tf], [x0, y0, z0], options);



% T is now a list of time values, while X is a matrix whose columns contain
% the numerical values of the x, y, and z variables.

% Note that, if we had not defined the function rhs above, we could include
% it as an inline function here by replacing the first argument with:
% @(T,X) [sigma*(X(2) - X(1)); X(1)*(rho - X(3)) - X(2); ...
%         X(1)*X(2) - beta*X(3)]
% This may sometimes be worthwhile if the ODE in question is very simple.

%=============================================================================
% Display various projections of the solution curve
%=============================================================================

% Note that, due to the chaotic nature of the Lorenz system, the numerical
% solutions from MATLAB and Python are expected to be slightly different.

figure % create a new figure window

% Show xy-projection
subplot(2, 2, 1) % target first (NW) tile in a 2x2 layout
plot(X(:,1), X(:,2)) % plot y versus x
xlabel("x")
ylabel("y")

% Show xz-projection
subplot(2, 2, 3) % target third (SW) tile in a 2x2 layout
plot(X(:,1), X(:,3)) % plot z versus x
xlabel("x")
ylabel("z")

% Show yz-projection
subplot(2, 2, 2) % target second (NE) tile in a 2x2 layout
plot(X(:,2), X(:,3)) % plot z versus y
xlabel("y")
ylabel("z")

% Show oblique projection
subplot(2, 2, 4) % target fourth (SE) tile in a 2x2 layout
plot3(X(:,1), X(:,2), X(:,3)) % plot 3D coordinates
xlabel("x")
ylabel("y")
zlabel("z")
view([225, 30]) % set view angle azimuth and elevation

% Add a title to an empty window
axes("visible", "off", "title", "The Lorenz Attractor")

% Note that, as of MATLAB R2018b, a subplot grid can be more easily titled by
% using the "sgtitle" function. This approach will work for older versions,
% and also for Octave.
