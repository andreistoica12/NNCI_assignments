epochs = 2; % number of epochs
P = 2; % number of sets
N = 2; % dimension of feature vectors

% we create P randomly-generated N-dimensional vectors
y = randn(P, N);

% x = randi([-1 1],10,1);

% we generate P random numbers with values either -1 or 1
S = 2*(rand(1,P)>0.5) - 1;

% intialize weights at 0
w = zeros(N, 1);

% double for-loop for the sequential perceptron training
for epoch = 1:epochs
    fprintf("epoch: %d\n", epoch);
    for example = 1:P % should probs rename this
        fprintf("P: %d\n", example);
        current_w = w;
        E = dot(current_w,(y(example,:).*S(example))); % not sure about which type of multiplication to do here
        fprintf("E = %d\n", E);
        if E <= 0
            w = current_w + (y(example,:).*S(example))'/N ; %I think
        end % dont need an else because the value doesn't change
    end
end
