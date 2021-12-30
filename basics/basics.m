% To include:
% Matrix-vector multiplication
% Vector-vector operations
% Vector-scalar operations
% Elementwise vector operations
% Solving a linear system
% Eigenvalues
% Determinant
% Generating data with linspace and creating a plot of a simple function

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

% Basic matrix operations
disp(det(B)); % Matrix determinant
disp(inv(B)); % Matrix inverse
disp(B'); % Matrix transpose
disp(B*A); % Matrix-matrix product
disp(B.*D); % Elementwise matrix-matrix product

% Defining vectors
u = rand(3, 1); % 3x1 random vector
v = [3; -4; 8]; % 3x1 vector defined elementwise
w = zeros(5, 1); % redefining a range of vector values
w(2:4) = 1;
x = linspace(0, 2*pi, 101); % vector of equally-spaced values on [0,2pi]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic programming structures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
% Different types of loop
% Different types of conditional
% Custom functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
% 2D and 3D plots of curves/surfaces
