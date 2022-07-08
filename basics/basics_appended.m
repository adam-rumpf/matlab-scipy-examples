                                                                                 % import numpy as np
                                                                                 % import matplotlib.pyplot as plt
                                                                                 % 
%=============================================================================   % #=============================================================================
% Basic linear algebra operations                                                % # Basic linear algebra operations
%=============================================================================   % #=============================================================================
                                                                                 % 
% Defining matrices                                                              % # Defining matrices
                                                                                 % 
A = rand(3, 2); % 3x2 random matrix                                              % A = np.random.rand(3, 2) # 3x2 random matrix
disp("A =");                                                                     % print("A =\n" + str(A))
disp(A);                                                                         % 
                                                                                 % 
B = [4, -3, 1; 9, 3, 2; -1, 0, 6]; % 3x3 matrix defined elementwise              % B = np.array([[4, -3, 1], [9, 3, 2], [-1, 0, 6]]) # 3x3 mat def elementwise
disp("\nB =");                                                                   % print("\nB =\n" + str(B))
disp(B);                                                                         % 
                                                                                 % 
C = eye(2); % 2x2 identity matrix                                                % C = np.eye(2) # 2x2 identity matrix
disp("\nC =");                                                                   % print("\nC =\n" + str(C))
disp(C);                                                                         % 
                                                                                 % 
D = zeros(3, 3); % 3x3 matrix filled using a loop                                % D = np.zeros((3, 3)) # 3x3 matrix filled using a loop
for i = 1:3                                                                      % for i in range(3):
    for j = 1:3                                                                  %     for j in range(3):
        D(i,j) = 3*(i-1) + j; % Note that MATLAB indices begin at 1              %         D[i][j] = 3*i + j + 1 # Note that Python indices begin at 0
    end                                                                          % print("\nD =\n" + str(D))
end                                                                              % 
disp("\nD =");                                                                   % 
disp(D);                                                                         % 
                                                                                 % 
E = [A B; C A']; % 5x5 matrix defined as a block matrix                          % E = np.block([[A, B], [C, A.T]]) # 5x5 matrix defined as a block matrix
disp("\nE =");                                                                   % print("\nE =\n" + str(E))
disp(E);                                                                         % 
                                                                                 % 
% Basic matrix operations                                                        % # Basic matrix operations
                                                                                 % 
disp("\ndet(B) =");                                                              % print("\ndet(B) =")
disp(det(B)); % Matrix determinant                                               % print(np.linalg.det(B)) # Matrix determinant
                                                                                 % 
disp("\ninv(B) =");                                                              % print("\ninv(B) =")
disp(inv(B)); % Matrix inverse                                                   % print(np.linalg.inv(B)) # Matrix inverse
                                                                                 % 
disp("\nB' =");                                                                  % print("\nB' =")
disp(B'); % Matrix transpose                                                     % print(B.T) # Matrix transpose
                                                                                 % 
disp("\n3B =");                                                                  % print("\n3B =")
disp(3*B); % Scalar multiplication                                               % print(3*B) # Scalar multiplication
                                                                                 % 
disp("\nBD =");                                                                  % print("\nBD =")
disp(B*D); % Matrix-matrix product                                               % print(B@D) # Matrix-matrix product
                                                                                 % 
disp("\nB.D =");                                                                 % print("\nB.D =")
disp(B.*D); % Elementwise matrix-matrix product                                  % print(B*D) # Elementwise matrix-matrix product
                                                                                 % 
% Defining vectors                                                               % # Defining vectors
                                                                                 % 
% Note that vectors in MATLAB are just matrices for which one dimension is 1,    % # Note that NumPy does not distinguish between column and row vectors, so
% so all matrix functions work with vectors.                                     % # anything that specifically requires a column or row vector requires
                                                                                 % # first converting the vector into a 2D array.
                                                                                 % 
u = rand(3, 1); % 3x1 random vector                                              % u = np.random.rand(3) # Length 3 random vector
disp("\nu =");                                                                   % print("\nu =\n" + str(u))
disp(u);                                                                         % 
                                                                                 % 
v = [3; -4; 8; 0; -1; 2]; % 6x1 vector defined elementwise                       % v = np.array([3, -4, 8, 0, -1, 2]) # Length 6 vector defined elementwise
disp("\nv =");                                                                   % print("\nv =\n" + str(v))
disp(v);                                                                         % 
                                                                                 % 
w = zeros(5, 1); % Redefining a range of vector values                           % w = np.zeros(5) # Redefining a range of vector values
w(2:4) = 1;                                                                      % w[1:4] = 1
disp("\nw =");                                                                   % print("\nw =\n" + str(w))
disp(w);                                                                         % 
                                                                                 % 
x = linspace(0, 2*pi, 101); % Row vector of equally-spaced values on [0,2pi]     % x = np.linspace(0, 2*np.pi, 101) # Vector of equally-spaced values on [0,2pi]
disp("\nx =");                                                                   % print("\nx =\n" + str(x[:11]) + "\n...\n" + str(x[-10:]))
disp(x(1:10));                                                                   % 
disp("...");                                                                     % 
disp(x(end-10:end));                                                             % 
                                                                                 % 
y = (10:-2:0)'; % Column vector of values defined by a range                     % y = np.array([i for i in range(10, -2, -2)]) # Values defined by a range
disp("\ny =");                                                                   % print("\ny =\n" + str(y))
disp(y);                                                                         % 
                                                                                 % 
% Basic vector operations                                                        % # Basic vector operations
                                                                                 % 
disp("\n2v =");                                                                  % print("\n2v =")
disp(2*v); % Scalar multiplication                                               % print(2*v) # Scalar multiplication
                                                                                 % 
disp("\nv'y =");                                                                 % print("\nv'y =")
disp(v'*y); % Inner product                                                      % print(np.inner(v, y)) # Inner product
                                                                                 % 
disp("\nvy' =");                                                                 % print("\nvy' =")
disp(v*y'); % Outer product                                                      % print(np.outer(v, y)) # Outer product
                                                                                 % 
disp("\nv.y =");                                                                 % print("\nv.y =")
disp(v.*y); % Elementwise vector-vector product                                  % print(v*y) # Elementwise vector-vector product
                                                                                 % 
disp("\nmin(v) =");                                                              % print("\nmin(v) =")
disp(min(v)); % Minimum value                                                    % print(min(v)) # Minimum value
                                                                                 % 
disp("\nmax(v) =");                                                              % print("\nmax(v) =")
disp(max(v)); % Maximum value                                                    % print(max(v)) # Maximum value
                                                                                 % 
disp("\nmean(v) =");                                                             % print("\nmean(v) =")
disp(mean(v)); % Average value                                                   % print(np.mean(v)) # Average value
                                                                                 % 
disp("\n||v||_2 =");                                                             % print("\n||v||_2 =")
disp(norm(v)); % Euclidean norm                                                  % print(np.linalg.norm(v)) # Euclidean norm
                                                                                 % 
disp("\n||v||_1 =");                                                             % print("\n||v||_1 =")
disp(norm(v, 1)); % Taxicab norm                                                 % print(np.linalg.norm(v, 1)) # Taxicab norm
                                                                                 % 
disp("\n||v||_inf =");                                                           % print("\n||v||_inf =")
disp(norm(v, Inf)); % Chebyshev norm                                             % print(np.linalg.norm(v, np.inf)) # Chebyshev norm
                                                                                 % 
% Eigenvalues                                                                    % # Eigenvalues
                                                                                 % 
% Note that MATLAB's eig() function can return either a vector of eigenvalues    % # Note that NumPy's eig() function always returns two arrays: A 2D array of
% or matrices of eigenvalues and eigenvectors depending on whether the output    % # eigenvectors, and a 1D array of eigenvalues, respectively. The separate
% is assigned to a single variable or a vector of two variables.                 % # eigvals() function returns only the eigenvalues.
                                                                                 % 
ew = eig(B); % Vector of eigenvalues                                             % ew = np.linalg.eigvals(B) # Vector of eigenvalues
disp("\nEigenvalues of B:");                                                     % print("\nEigenvalues of B:\n" + str(ew))
disp(ew);                                                                        % 
                                                                                 % 
[ev, ew] = eig(B); % Matrices of eigenvalues and eigenvectors                    % ew, ev = np.linalg.eig(B) # Matrices of eigenvalues and eigenvectors
disp("\nEigenvectors of B:");                                                    % print("\nEigenvectors of B:\n" + str(ev))
disp(ev);                                                                        % 
                                                                                 % 
disp("\nX Lambda X^(-1) =");                                                     % print("\nX Lambda X^(-1) =")
disp(ev*ew*inv(ev)); % Demonstrating the eigendecomposition of B                 % print(ev @ np.diag(ew) @ np.linalg.inv(ev)) # Eigendecomposition of B
                                                                                 % 
% Linear systems                                                                 % # Linear systems
                                                                                 % 
v = v(1:3);                                                                      % v = v[:3]
z = B \ v; % Solve the linear system Bz = v                                      % z = np.linalg.solve(B, v) # Solve the linear system Bz = v
disp("\nz =");                                                                   % print("\nz =\n" + str(z))
disp(z);                                                                         % 
                                                                                 % 
disp("\nBz =");                                                                  % print("\nBz =")
disp(B*z); % Matrix-vector product                                               % print(B@z) # Matrix-vector product
                                                                                 % 
%=============================================================================   % #=============================================================================
% Basic programming structures                                                   % # Basic programming structures
%=============================================================================   % #=============================================================================
                                                                                 % 
% Conditionals                                                                   % # Conditionals
                                                                                 % 
if det(B) > 0                                                                    % if np.linalg.det(B) > 0:
    disp("\ndet(B) is positive");                                                %     print("\ndet(B) is positive")
elseif det(B) < 0                                                                % elif np.linalg.det(B) < 0:
    disp("\ndet(B) is negative");                                                %     print("\ndet(B) is negative")
else                                                                             % else:
    disp("\ndet(B) is zero");                                                    %     print("\ndet(B) is zero")
end                                                                              % 
                                                                                 % 
% For loops                                                                      % # For loops
                                                                                 % 
s = zeros(1, 10);                                                                % s = np.zeros(10)
for i = 1:length(s)                                                              % for i in range(len(s)):
    s(i) = sqrt(i); % Square root                                                %     s[i] = np.sqrt(i+1) # Square root
end                                                                              % print("\ns =\n" + str(s))
disp("\ns =");                                                                   % 
disp(s);                                                                         % 
                                                                                 % 
% While loops                                                                    % # While loops
                                                                                 % 
p = 0;                                                                           % p = 0
while 2^p < 1000                                                                 % while 2**p < 1000:
    p += 1; % Combined assignment                                                %     p += 1 # Combined assignment
end                                                                              % print("\np =\n" + str(p))
disp("\np =");                                                                   % 
disp(p);                                                                         % 
                                                                                 % 
% Functions                                                                      % # Functions
                                                                                 % 
function out = square(x)                                                         % def square(x):
    out = x.^2;                                                                  %     return x**2
end                                                                              % 
                                                                                 % print("\nsquare(y) =\n" + str(square(y)))
disp("\nsquare(y) =");                                                           % 
disp(square(y));                                                                 % 
                                                                                 % 
%=============================================================================   % #=============================================================================
% Basic plotting                                                                 % # Basic plotting
%=============================================================================   % #=============================================================================
                                                                                 % 
% Single 2D plot                                                                 % # Single 2D plot
                                                                                 % 
% In MATLAB, multiple series can be plotted on the same axis either by passing   % # In Pyplot, multiple series can be plotted on the same axis either by passing
% all of them to the plot() function, or by toggling 'hold' to 'on' and then     % # all of them to the plot() function, or plotting them separately before using
% plotting them separately.                                                      % # the show() function.
% Series labels in the legend are most easily added by specifying them, in       % # Series labels in the legend are most easily added by defining them in the
% order, in the legend() function.                                               % # 'label' keyword argument of the respective plot() function before calling the
                                                                                 % # legend() function.
                                                                                 % 
y1 = cos(x); % Generate function values on [0,2pi]                               % y1 = np.cos(x) # Generate function values on [0,2pi]
y2 = sin(x);                                                                     % y2 = np.sin(x)
                                                                                 % 
figure % Create a new figure window                                              % plt.figure() # Create a new figure window
hold on % Turn hold on to collect multiple plots in the same figure              % 
plot(x, y1, 'r') % Plot a red solid line                                         % plt.plot(x, y1, 'r', label="cos(x)") # Plot a red solid line
plot(x, y2, 'b.') % Plot a blue dotted line                                      % plt.plot(x, y2, 'b.', label="sin(x)") # Plot a blue dotted line
title("Example 1") % Figure title                                                % plt.title("Example 1") # Figure title
xlabel("x") % x-axis label                                                       % plt.xlabel("x") # x-axis label
ylabel("y") % y-axis label                                                       % plt.ylabel("y") # y-axis label
legend("cos(x)", "sin(x)") % Plot legend                                         % plt.legend() # Plot legend (automatically adopts legend keywords from plots)
hold off                                                                         % 
                                                                                 % 
% Stacked 2D plots                                                               % # Stacked 2D plots
                                                                                 % 
figure                                                                           % plt.figure()
subplot(2, 1, 1) % Target first tile in a 2x1 layout                             % plt.subplot(2, 1, 1) # Target first tile in a 2x1 layout
plot(x, y1, 'r') % Plot first function                                           % plt.plot(x, y1, 'r') # Plot first function
title("Example 2") % Add title above first subplot                               % plt.title("Example 2") # Add title above first subplot
subplot(2, 1, 2) % Target second tile in a 2x1 layout                            % plt.subplot(2, 1, 2) # Target second tile in a 2x1 layout
plot(x, y2, 'b.') % Plot second function                                         % plt.plot(x, y2, 'b.') # Plot second function
                                                                                 % 
% 3D surface plots                                                               % # 3D surface plots
                                                                                 % 
% Surface plots are usually generated with the help of MATLAB's meshgrid()       % # Surface plots are usually generated with the help of NumPy's meshgrid()
% function, which returns a pair of matrices: One whose rows are identical       % # function, which returns a pair of matrices: One whose rows are identical
% copies of the x-values, and one whose columns are identical copies of the      % # copies of the x-values, and one whose columns are identical copies of the
% y-values. The mesh of z-coordinates can then be computed by evaluating an      % # y-values. The mesh of z-coordinates can then be computed by evaluating an
% elementwise function with these two matrices as the x- and y-values.           % # elementwise function with these two matrices as the x- and y-values.
                                                                                 % 
[X, Y] = meshgrid(linspace(-4,4,41), linspace(-4,4,41)); % Generate grids        % [X, Y] = np.meshgrid(np.linspace(-4,4,41), np.linspace(-4,4,41)) # Grids
                                                                                 % 
function z = f(x, y) % Define a function for the surface                         % def f(x, y): # Define a function for the surface
    z = 100*cos(x) + 1 - 10*y.^2 + y.^4;                                         %     return 100*np.cos(x) + 1 - 10*y**2 + y**4
end                                                                              % 
                                                                                 % 
Z = f(X, Y); % Generate z-coordinates                                            % Z = f(X, Y) # Generate z-coordinates
                                                                                 % 
figure                                                                           % plt.figure()
                                                                                 % ax = plt.axes(projection='3d') # Get handle for 3D axis in current figure
surf(X, Y, Z) % Surface plot                                                     % ax.plot_surface(X, Y, Z) # Surface plot
title("Example 3")                                                               % plt.title("Example 3")
                                                                                 % 
                                                                                 % plt.show() # Show all generated figures