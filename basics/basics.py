import numpy as np

#=============================================================================
# Basic linear algebra operations
#=============================================================================

# Defining matrices

A = np.random.rand(3, 2) # 3x2 random matrix
print("A =\n" + str(A))


B = np.array([[4, -3, 1], [9, 3, 2], [-1, 0, 6]]) # 3x3 mat def elementwise
print("\nB =\n" + str(B))


C = np.eye(2) # 2x2 identity matrix
print("\nC =\n" + str(C))


D = np.zeros((3, 3)) # 3x3 matrix filled using a loop
for i in range(3):
    for j in range(3):
        D[i][j] = 3*i + j + 1 # Note that Python indices begin at 0
print("\nD =\n" + str(D))




E = np.block([[A, B], [C, A.T]]) # 5x5 matrix defined as a block matrix
print("\nE =\n" + str(E))


# Basic matrix operations

print("\ndet(B) =")
print(np.linalg.det(B)) # Matrix determinant

print("\ninv(B) =")
print(np.linalg.inv(B)) # Matrix inverse

print("\nB' =")
print(B.T) # Matrix transpose

print("\n3B =")
print(3*B) # Scalar multiplication

print("\nBD =")
print(B@D) # Matrix-matrix product

print("\nB.D =")
print(B*D) # Elementwise matrix-matrix product

# Defining vectors

# Note that NumPy does not distinguish between column and row vectors, so
# anything that specifically requires a column or row vector requires
# first converting the vector into a 2D array.

u = np.random.rand(3) # Length 3 random vector
print("\nu =\n" + str(u))


v = np.array([3, -4, 8, 0, -1, 2]) # Length 6 vector defined elementwise
print("\nv =\n" + str(v))


w = np.zeros(5) # Redefining a range of vector values
w[1:4] = 1
print("\nw =\n" + str(w))


x = np.linspace(0, 2*np.pi, 101) # Vector of equally-spaced values on [0,2pi]
print("\nx =\n" + str(x[:11]) + "\n...\n" + str(x[-10:]))




y = np.array([i for i in range(10, -2, -2)]) # Vector of vals defined by a ran
print("\ny =\n" + str(y))


# Basic vector operations

print("\n2v =")
print(2*v) # Scalar multiplication

print("\nv'y =")
print(np.inner(v, y)) # Inner product

print("\nvy' =")
print(np.outer(v, y)) # Outer product

print("\nv.y =")
print(v*y) # Elementwise vector-vector product

print("\nmin(v) =")
print(min(v)) # Minimum value

print("\nmax(v) =")
print(max(v)) # Maximum value

print("\nmean(v) =")
print(np.mean(v)) # Average value

print("\n||v||_2 =")
print(np.linalg.norm(v)) # Euclidean norm

print("\n||v||_1 =")
print(np.linalg.norm(v, 1)) # Taxicab norm

print("\n||v||_inf =")
print(np.linalg.norm(v, np.inf)) # Chebyshev norm

# Eigenvalues

# Note that NumPy's eig() function always returns two arrays: A 2D array of
# eigenvectors, and a 1D array of eigenvalues, respectively. The separate
# eigvals() function returns only the eigenvalues.

ew = np.linalg.eigvals(B) # Vector of eigenvalues
print("\nEigenvalues of B:\n" + str(ew))


ew, ev = np.linalg.eig(B) # Matrices of eigenvalues and eigenvectors
print("\nEigenvectors of B:\n" + str(ev))


print("\nX Lambda X^(-1) =")
print(ev@np.diag(ew)@np.linalg.inv(ev)) # Demonstrating the eigendecomposition of B

# Linear systems

v = v[:3]
z = np.linalg.solve(B, v) # Solve the linear system Bz = v
print("\nz =\n" + str(z))


print("\nBz =")
print(B@z) # Matrix-vector product

#=============================================================================
# Basic programming structures
#=============================================================================

# Conditionals

if np.linalg.det(B) > 0:
    print("\ndet(B) is positive")
elif np.linalg.det(B) < 0:
    print("\ndet(B) is negative")
else:
    print("\ndet(B) is zero")


# For loops

s = np.zeros(10)
for i in range(len(s)):
    s[i] = np.sqrt(i+1) # Square root
print("\ns =\n" + str(s))



# While loops

p = 0
while 2**p < 1000:
    p += 1 # Combined assignment
print("\np =\n" + str(p))



# Functions

def square(x):
    return x**2

print("\nsquare(y) =\n" + str(square(y)))



#=============================================================================
# Basic plotting
#=============================================================================

###
# 2D and 3D plots of curves/surfaces
# Generating data with linspace and creating a plot of a simple function
