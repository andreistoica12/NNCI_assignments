P = 20;
N = 2;

% we create P randomly-generated N-dimensional vectors
y = randn(P, N);

% x = randi([-1 1],10,1);

% we generate P random numbers with values either -1 or 1
S = 2*(rand(1,P)>0.5) - 1;