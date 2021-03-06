


%=============================================================================
% Basic linear algebra operations
%=============================================================================

% Defining matrices

A = rand(3, 2); % 3x2 random matrix
disp("A =");
disp(A);

B = [4, -3, 1; 9, 3, 2; -1, 0, 6]; % 3x3 matrix defined elementwise
disp("\nB =");
disp(B);

C = eye(2); % 2x2 identity matrix
disp("\nC =");
disp(C);

D = zeros(3, 3); % 3x3 matrix filled using a loop
for i = 1:3
	for j = 1:3
		D(i,j) = 3*(i-1) + j; % Note that MATLAB indices begin at 1
	end
end
disp("\nD =");
disp(D);

E = [A B; C A']; % 5x5 matrix defined as a block matrix
disp("\nE =");
disp(E);

% Basic matrix operations

disp("\ndet(B) =");
disp(det(B)); % Matrix determinant

disp("\ninv(B) =");
disp(inv(B)); % Matrix inverse

disp("\nB' =");
disp(B'); % Matrix transpose

disp("\n3B =");
disp(3*B); % Scalar multiplication

disp("\nBD =");
disp(B*D); % Matrix-matrix product

disp("\nB.D =");
disp(B.*D); % Elementwise matrix-matrix product

% Defining vectors

% Note that vectors in MATLAB are just matrices for which one dimension is 1,
% so all matrix functions work with vectors.


u = rand(3, 1); % 3x1 random vector
disp("\nu =");
disp(u);

v = [3; -4; 8; 0; -1; 2]; % 6x1 vector defined elementwise
disp("\nv =");
disp(v);

w = zeros(5, 1); % Redefining a range of vector values
w(2:4) = 1;
disp("\nw =");
disp(w);

x = linspace(0, 2*pi, 101); % Row vector of equally-spaced values on [0,2pi]
disp("\nx =");
disp(x(1:10));
disp("...");
disp(x(end-10:end));

y = (10:-2:0)'; % Column vector of values defined by a range
disp("\ny =");
disp(y);

% Basic vector operations

disp("\n2v =");
disp(2*v); % Scalar multiplication

disp("\nv'y =");
disp(v'*y); % Inner product

disp("\nvy' =");
disp(v*y'); % Outer product

disp("\nv.y =");
disp(v.*y); % Elementwise vector-vector product

disp("\nmin(v) =");
disp(min(v)); % Minimum value

disp("\nmax(v) =");
disp(max(v)); % Maximum value

disp("\nmean(v) =");
disp(mean(v)); % Average value

disp("\n||v||_2 =");
disp(norm(v)); % Euclidean norm

disp("\n||v||_1 =");
disp(norm(v, 1)); % Taxicab norm

disp("\n||v||_inf =");
disp(norm(v, Inf)); % Chebyshev norm

% Eigenvalues

% Note that MATLAB's eig() function can return either a vector of eigenvalues
% or matrices of eigenvalues and eigenvectors depending on whether the output
% is assigned to a single variable or a vector of two variables.

ew = eig(B); % Vector of eigenvalues
disp("\nEigenvalues of B:");
disp(ew);

[ev, ew] = eig(B); % Matrices of eigenvalues and eigenvectors
disp("\nEigenvectors of B:");
disp(ev);

disp("\nX Lambda X^(-1) =");
disp(ev*ew*inv(ev)); % Demonstrating the eigendecomposition of B

% Linear systems

v = v(1:3);
z = B \ v; % Solve the linear system Bz = v
disp("\nz =");
disp(z);

disp("\nBz =");
disp(B*z); % Matrix-vector product

%=============================================================================
% Basic programming structures
%=============================================================================

% Conditionals

if det(B) > 0
	disp("\ndet(B) is positive");
elseif det(B) < 0
	disp("\ndet(B) is negative");
else
	disp("\ndet(B) is zero");
end

% For loops

s = zeros(1, 10);
for i = 1:length(s)
	s(i) = sqrt(i); % Square root
end
disp("\ns =");
disp(s);

% While loops

p = 0;
while 2^p < 1000
	p += 1; % Combined assignment
end
disp("\np =");
disp(p);

% Functions

function out = square(x)
	out = x.^2;
end

disp("\nsquare(y) =");
disp(square(y));

%=============================================================================
% Basic plotting
%=============================================================================

% Single 2D plot

% In MATLAB, multiple series can be plotted on the same axis either by passing
% all of them to the plot() function, or by toggling 'hold' to 'on' and then
% plotting them separately.
% Series labels in the legend are most easily added by specifying them, in
% order, in the legend() function.


y1 = cos(x); % Generate function values on [0,2pi]
y2 = sin(x);

figure % Create a new figure window
hold on % Turn hold on to collect multiple plots in the same figure
plot(x, y1, 'r') % Plot a red solid line
plot(x, y2, 'b.') % Plot a blue dotted line
title("Example 1") % Figure title
xlabel("x") % x-axis label
ylabel("y") % y-axis label
legend("cos(x)", "sin(x)") % Plot legend
hold off

% Stacked 2D plots

figure
subplot(2, 1, 1) % Target first tile in a 2x1 layout
plot(x, y1, 'r') % Plot first function
title("Example 2") % Add title above first subplot
subplot(2, 1, 2) % Target second tile in a 2x1 layout
plot(x, y2, 'b.') % Plot second function

% 3D surface plots

% Surface plots are usually generated with the help of MATLAB's meshgrid()
% function, which returns a pair of matrices: One whose rows are identical
% copies of the x-values, and one whose columns are identical copies of the
% y-values. The mesh of z-coordinates can then be computed by evaluating an
% elementwise function with these two matrices as the x- and y-values.

[X, Y] = meshgrid(linspace(-4,4,41), linspace(-4,4,41)); % Generate grids

function z = f(x, y) % Define a function for the surface
	z = 100*cos(x) + 1 - 10*y.^2 + y.^4;
end

Z = f(X, Y); % Generate z-coordinates

figure

surf(X, Y, Z) % Surface plot
title("Example 3")


