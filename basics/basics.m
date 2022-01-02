%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic linear algebra operations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Defining matrices

A = rand(3, 2); % 3x2 random matrix

B = [4, -3, 1; 9, 3, 2; -1, 0, 6]; % 3x3 matrix defined elementwise

C = eye(2); % 2x2 identity matrix

D = zeros(3, 3); % 3x3 matrix filled using a loop
for i = 1:3
	for j = 1:3
		D(i,j) = 3*(i-1) + j;
	end
end

E = [A B; C A']; % 5x5 matrix defined as a block matrix

% Basic matrix operations

disp(det(B)); % Matrix determinant

disp(inv(B)); % Matrix inverse

disp(B'); % Matrix transpose

disp(3*B); % Scalar multiplication

disp(B*A); % Matrix-matrix product

disp(B.*D); % Elementwise matrix-matrix product

% Defining vectors

u = rand(3, 1); % 3x1 random vector

v = [3; -4; 8]; % 3x1 vector defined elementwise

w = zeros(5, 1); % Redefining a range of vector values
w(2:4) = 1;

x = linspace(0, 2*pi, 101); % Vector of equally-spaced values on [0,2pi]

y = 10:-2:0; % Vector of values defined by a range

% Basic vector operations

disp(2*v); % Scalar multiplication

disp(u'*v); % Inner product

disp(u*v'); % Outer product

disp(u.*v); % Elementwise vector-vector product

disp(min(v)); % Minimum value

disp(max(v)); % Maximum value

disp(mean(v)); % Average value

disp(norm(v)); % Euclidean norm

disp(norm(v, 1)); % Taxicab norm

disp(norm(v, Inf)); % Chebyshev norm

% Eigenvalues

ew = eig(B); % Vector of eigenvalues
disp(ew);

[ev, ew] = eig(B); % Matrices of eigenvalues and eigenvectors
disp(ev);

disp(ev*ew*inv(ev)); % Demonstrating the eigendecomposition of B

% Linear systems

z = B \ v; % Solve the linear system Bz = v

disp(B*z); % Matrix-vector product

disp(rref([B v])); % Reduced row echelon form of the augmented matrix [B|v]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic programming structures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Conditionals

if det(D) > 0
	disp("det(D) is positive");
elseif det(D) < 0
	disp("det(D) is negative");
else
	disp("det(D) is zero");
end

% For loops

s = zeros(1, 10);
for i = 1:10
	s(i) = sqrt(i); % Square root
end

% While loops

p = 0;
while 2^p < 1000
	p += 1; % Combined assignment
end

% Functions

function out = square(x)
	out = x.^2;
end

disp(square(y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
% 2D and 3D plots of curves/surfaces
% Generating data with linspace and creating a plot of a simple function
