import numpy as np                                                               # 
import matplotlib.pyplot as plt                                                  # 
                                                                                 # 
#=============================================================================   # %=============================================================================
# Basic linear algebra operations                                                # % Basic linear algebra operations
#=============================================================================   # %=============================================================================
                                                                                 # 
# Defining matrices                                                              # % Defining matrices
                                                                                 # 
A = np.random.rand(3, 2) # 3x2 random matrix                                     # A = rand(3, 2); % 3x2 random matrix
print("A =\n" + str(A))                                                          # disp("A =");
                                                                                 # disp(A);
                                                                                 # 
B = np.array([[4, -3, 1], [9, 3, 2], [-1, 0, 6]]) # 3x3 mat def elementwise      # B = [4, -3, 1; 9, 3, 2; -1, 0, 6]; % 3x3 matrix defined elementwise
print("\nB =\n" + str(B))                                                        # disp("\nB =");
                                                                                 # disp(B);
                                                                                 # 
C = np.eye(2) # 2x2 identity matrix                                              # C = eye(2); % 2x2 identity matrix
print("\nC =\n" + str(C))                                                        # disp("\nC =");
                                                                                 # disp(C);
                                                                                 # 
D = np.zeros((3, 3)) # 3x3 matrix filled using a loop                            # D = zeros(3, 3); % 3x3 matrix filled using a loop
for i in range(3):                                                               # for i = 1:3
    for j in range(3):                                                           #     for j = 1:3
        D[i][j] = 3*i + j + 1 # Note that Python indices begin at 0              #         D(i,j) = 3*(i-1) + j; % Note that MATLAB indices begin at 1
print("\nD =\n" + str(D))                                                        #     end
                                                                                 # end
                                                                                 # disp("\nD =");
                                                                                 # disp(D);
                                                                                 # 
E = np.block([[A, B], [C, A.T]]) # 5x5 matrix defined as a block matrix          # E = [A B; C A']; % 5x5 matrix defined as a block matrix
print("\nE =\n" + str(E))                                                        # disp("\nE =");
                                                                                 # disp(E);
                                                                                 # 
# Basic matrix operations                                                        # % Basic matrix operations
                                                                                 # 
print("\ndet(B) =")                                                              # disp("\ndet(B) =");
print(np.linalg.det(B)) # Matrix determinant                                     # disp(det(B)); % Matrix determinant
                                                                                 # 
print("\ninv(B) =")                                                              # disp("\ninv(B) =");
print(np.linalg.inv(B)) # Matrix inverse                                         # disp(inv(B)); % Matrix inverse
                                                                                 # 
print("\nB' =")                                                                  # disp("\nB' =");
print(B.T) # Matrix transpose                                                    # disp(B'); % Matrix transpose
                                                                                 # 
print("\n3B =")                                                                  # disp("\n3B =");
print(3*B) # Scalar multiplication                                               # disp(3*B); % Scalar multiplication
                                                                                 # 
print("\nBD =")                                                                  # disp("\nBD =");
print(B@D) # Matrix-matrix product                                               # disp(B*D); % Matrix-matrix product
                                                                                 # 
print("\nB.D =")                                                                 # disp("\nB.D =");
print(B*D) # Elementwise matrix-matrix product                                   # disp(B.*D); % Elementwise matrix-matrix product
                                                                                 # 
# Defining vectors                                                               # % Defining vectors
                                                                                 # 
# Note that NumPy does not distinguish between column and row vectors, so        # % Note that vectors in MATLAB are just matrices for which one dimension is 1,
# anything that specifically requires a column or row vector requires            # % so all matrix functions work with vectors.
# first converting the vector into a 2D array.                                   # 
                                                                                 # 
u = np.random.rand(3) # Length 3 random vector                                   # u = rand(3, 1); % 3x1 random vector
print("\nu =\n" + str(u))                                                        # disp("\nu =");
                                                                                 # disp(u);
                                                                                 # 
v = np.array([3, -4, 8, 0, -1, 2]) # Length 6 vector defined elementwise         # v = [3; -4; 8; 0; -1; 2]; % 6x1 vector defined elementwise
print("\nv =\n" + str(v))                                                        # disp("\nv =");
                                                                                 # disp(v);
                                                                                 # 
w = np.zeros(5) # Redefining a range of vector values                            # w = zeros(5, 1); % Redefining a range of vector values
w[1:4] = 1                                                                       # w(2:4) = 1;
print("\nw =\n" + str(w))                                                        # disp("\nw =");
                                                                                 # disp(w);
                                                                                 # 
x = np.linspace(0, 2*np.pi, 101) # Vector of equally-spaced values on [0,2pi]    # x = linspace(0, 2*pi, 101); % Row vector of equally-spaced values on [0,2pi]
print("\nx =\n" + str(x[:11]) + "\n...\n" + str(x[-10:]))                        # disp("\nx =");
                                                                                 # disp(x(1:10));
                                                                                 # disp("...");
                                                                                 # disp(x(end-10:end));
                                                                                 # 
y = np.array([i for i in range(10, -2, -2)]) # Values defined by a range         # y = (10:-2:0)'; % Column vector of values defined by a range
print("\ny =\n" + str(y))                                                        # disp("\ny =");
                                                                                 # disp(y);
                                                                                 # 
# Basic vector operations                                                        # % Basic vector operations
                                                                                 # 
print("\n2v =")                                                                  # disp("\n2v =");
print(2*v) # Scalar multiplication                                               # disp(2*v); % Scalar multiplication
                                                                                 # 
print("\nv'y =")                                                                 # disp("\nv'y =");
print(np.inner(v, y)) # Inner product                                            # disp(v'*y); % Inner product
                                                                                 # 
print("\nvy' =")                                                                 # disp("\nvy' =");
print(np.outer(v, y)) # Outer product                                            # disp(v*y'); % Outer product
                                                                                 # 
print("\nv.y =")                                                                 # disp("\nv.y =");
print(v*y) # Elementwise vector-vector product                                   # disp(v.*y); % Elementwise vector-vector product
                                                                                 # 
print("\nmin(v) =")                                                              # disp("\nmin(v) =");
print(min(v)) # Minimum value                                                    # disp(min(v)); % Minimum value
                                                                                 # 
print("\nmax(v) =")                                                              # disp("\nmax(v) =");
print(max(v)) # Maximum value                                                    # disp(max(v)); % Maximum value
                                                                                 # 
print("\nmean(v) =")                                                             # disp("\nmean(v) =");
print(np.mean(v)) # Average value                                                # disp(mean(v)); % Average value
                                                                                 # 
print("\n||v||_2 =")                                                             # disp("\n||v||_2 =");
print(np.linalg.norm(v)) # Euclidean norm                                        # disp(norm(v)); % Euclidean norm
                                                                                 # 
print("\n||v||_1 =")                                                             # disp("\n||v||_1 =");
print(np.linalg.norm(v, 1)) # Taxicab norm                                       # disp(norm(v, 1)); % Taxicab norm
                                                                                 # 
print("\n||v||_inf =")                                                           # disp("\n||v||_inf =");
print(np.linalg.norm(v, np.inf)) # Chebyshev norm                                # disp(norm(v, Inf)); % Chebyshev norm
                                                                                 # 
# Eigenvalues                                                                    # % Eigenvalues
                                                                                 # 
# Note that NumPy's eig() function always returns two arrays: A 2D array of      # % Note that MATLAB's eig() function can return either a vector of eigenvalues
# eigenvectors, and a 1D array of eigenvalues, respectively. The separate        # % or matrices of eigenvalues and eigenvectors depending on whether the output
# eigvals() function returns only the eigenvalues.                               # % is assigned to a single variable or a vector of two variables.
                                                                                 # 
ew = np.linalg.eigvals(B) # Vector of eigenvalues                                # ew = eig(B); % Vector of eigenvalues
print("\nEigenvalues of B:\n" + str(ew))                                         # disp("\nEigenvalues of B:");
                                                                                 # disp(ew);
                                                                                 # 
ew, ev = np.linalg.eig(B) # Matrices of eigenvalues and eigenvectors             # [ev, ew] = eig(B); % Matrices of eigenvalues and eigenvectors
print("\nEigenvectors of B:\n" + str(ev))                                        # disp("\nEigenvectors of B:");
                                                                                 # disp(ev);
                                                                                 # 
print("\nX Lambda X^(-1) =")                                                     # disp("\nX Lambda X^(-1) =");
print(ev @ np.diag(ew) @ np.linalg.inv(ev)) # Eigendecomposition of B            # disp(ev*ew*inv(ev)); % Demonstrating the eigendecomposition of B
                                                                                 # 
# Linear systems                                                                 # % Linear systems
                                                                                 # 
v = v[:3]                                                                        # v = v(1:3);
z = np.linalg.solve(B, v) # Solve the linear system Bz = v                       # z = B \ v; % Solve the linear system Bz = v
print("\nz =\n" + str(z))                                                        # disp("\nz =");
                                                                                 # disp(z);
                                                                                 # 
print("\nBz =")                                                                  # disp("\nBz =");
print(B@z) # Matrix-vector product                                               # disp(B*z); % Matrix-vector product
                                                                                 # 
#=============================================================================   # %=============================================================================
# Basic programming structures                                                   # % Basic programming structures
#=============================================================================   # %=============================================================================
                                                                                 # 
# Conditionals                                                                   # % Conditionals
                                                                                 # 
if np.linalg.det(B) > 0:                                                         # if det(B) > 0
    print("\ndet(B) is positive")                                                #     disp("\ndet(B) is positive");
elif np.linalg.det(B) < 0:                                                       # elseif det(B) < 0
    print("\ndet(B) is negative")                                                #     disp("\ndet(B) is negative");
else:                                                                            # else
    print("\ndet(B) is zero")                                                    #     disp("\ndet(B) is zero");
                                                                                 # end
                                                                                 # 
# For loops                                                                      # % For loops
                                                                                 # 
s = np.zeros(10)                                                                 # s = zeros(1, 10);
for i in range(len(s)):                                                          # for i = 1:length(s)
    s[i] = np.sqrt(i+1) # Square root                                            #     s(i) = sqrt(i); % Square root
print("\ns =\n" + str(s))                                                        # end
                                                                                 # disp("\ns =");
                                                                                 # disp(s);
                                                                                 # 
# While loops                                                                    # % While loops
                                                                                 # 
p = 0                                                                            # p = 0;
while 2**p < 1000:                                                               # while 2^p < 1000
    p += 1 # Combined assignment                                                 #     p += 1; % Combined assignment
print("\np =\n" + str(p))                                                        # end
                                                                                 # disp("\np =");
                                                                                 # disp(p);
                                                                                 # 
# Functions                                                                      # % Functions
                                                                                 # 
def square(x):                                                                   # function out = square(x)
    return x**2                                                                  #     out = x.^2;
                                                                                 # end
print("\nsquare(y) =\n" + str(square(y)))                                        # 
                                                                                 # disp("\nsquare(y) =");
                                                                                 # disp(square(y));
                                                                                 # 
#=============================================================================   # %=============================================================================
# Basic plotting                                                                 # % Basic plotting
#=============================================================================   # %=============================================================================
                                                                                 # 
# Single 2D plot                                                                 # % Single 2D plot
                                                                                 # 
# In Pyplot, multiple series can be plotted on the same axis either by passing   # % In MATLAB, multiple series can be plotted on the same axis either by passing
# all of them to the plot() function, or plotting them separately before using   # % all of them to the plot() function, or by toggling 'hold' to 'on' and then
# the show() function.                                                           # % plotting them separately.
# Series labels in the legend are most easily added by defining them in the      # % Series labels in the legend are most easily added by specifying them, in
# 'label' keyword argument of the respective plot() function before calling the  # % order, in the legend() function.
# legend() function.                                                             # 
                                                                                 # 
y1 = np.cos(x) # Generate function values on [0,2pi]                             # y1 = cos(x); % Generate function values on [0,2pi]
y2 = np.sin(x)                                                                   # y2 = sin(x);
                                                                                 # 
plt.figure() # Create a new figure window                                        # figure % Create a new figure window
                                                                                 # hold on % Turn hold on to collect multiple plots in the same figure
plt.plot(x, y1, 'r', label="cos(x)") # Plot a red solid line                     # plot(x, y1, 'r') % Plot a red solid line
plt.plot(x, y2, 'b.', label="sin(x)") # Plot a blue dotted line                  # plot(x, y2, 'b.') % Plot a blue dotted line
plt.title("Example 1") # Figure title                                            # title("Example 1") % Figure title
plt.xlabel("x") # x-axis label                                                   # xlabel("x") % x-axis label
plt.ylabel("y") # y-axis label                                                   # ylabel("y") % y-axis label
plt.legend() # Plot legend (automatically adopts legend keywords from plots)     # legend("cos(x)", "sin(x)") % Plot legend
                                                                                 # hold off
                                                                                 # 
# Stacked 2D plots                                                               # % Stacked 2D plots
                                                                                 # 
plt.figure()                                                                     # figure
plt.subplot(2, 1, 1) # Target first tile in a 2x1 layout                         # subplot(2, 1, 1) % Target first tile in a 2x1 layout
plt.plot(x, y1, 'r') # Plot first function                                       # plot(x, y1, 'r') % Plot first function
plt.title("Example 2") # Add title above first subplot                           # title("Example 2") % Add title above first subplot
plt.subplot(2, 1, 2) # Target second tile in a 2x1 layout                        # subplot(2, 1, 2) % Target second tile in a 2x1 layout
plt.plot(x, y2, 'b.') # Plot second function                                     # plot(x, y2, 'b.') % Plot second function
                                                                                 # 
# 3D surface plots                                                               # % 3D surface plots
                                                                                 # 
# Surface plots are usually generated with the help of NumPy's meshgrid()        # % Surface plots are usually generated with the help of MATLAB's meshgrid()
# function, which returns a pair of matrices: One whose rows are identical       # % function, which returns a pair of matrices: One whose rows are identical
# copies of the x-values, and one whose columns are identical copies of the      # % copies of the x-values, and one whose columns are identical copies of the
# y-values. The mesh of z-coordinates can then be computed by evaluating an      # % y-values. The mesh of z-coordinates can then be computed by evaluating an
# elementwise function with these two matrices as the x- and y-values.           # % elementwise function with these two matrices as the x- and y-values.
                                                                                 # 
[X, Y] = np.meshgrid(np.linspace(-4,4,41), np.linspace(-4,4,41)) # Grids         # [X, Y] = meshgrid(linspace(-4,4,41), linspace(-4,4,41)); % Generate grids
                                                                                 # 
def f(x, y): # Define a function for the surface                                 # function z = f(x, y) % Define a function for the surface
    return 100*np.cos(x) + 1 - 10*y**2 + y**4                                    #     z = 100*cos(x) + 1 - 10*y.^2 + y.^4;
                                                                                 # end
                                                                                 # 
Z = f(X, Y) # Generate z-coordinates                                             # Z = f(X, Y); % Generate z-coordinates
                                                                                 # 
plt.figure()                                                                     # figure
ax = plt.axes(projection='3d') # Get handle for 3D axis in current figure        # 
ax.plot_surface(X, Y, Z) # Surface plot                                          # surf(X, Y, Z) % Surface plot
plt.title("Example 3")                                                           # title("Example 3")
                                                                                 # 
plt.show() # Show all generated figures                                          # 