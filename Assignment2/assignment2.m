epochs = 100; % number of epochs
P = 5;
N = 5;
% we create P randomly-generated N-dimensional vectors
y = randn(P, N);

S = zeros(P, 1);
w_star = randn(N, 1);
w_star = norm(w_star) * sqrt(N); %  normalise to 1 then multiply to the preferred value sqrt(N)

for item = 1:P
    temp = w_star * y(item, :);
    if temp < 0
        S(item) = -1;
    else
        S(item) = 1;
    end 
end 

w = zeros(N, 1);
k = zeros(N, 1);
E = zeros(N, 1);

for epoch = 1:epochs
    for example = 1:P
        current_w = w;
        E(example) = dot(current_w, y(example, :).*S(example));
        if norm(current_w) ~= 0
            k(example) = E(example) ./ norm(current_w);
        else
            k(example) = E(example);  %? 
        end
    end
    % we determine the example with lowest stability
    [min_example, index] = min(k); % find(k == min(k))
    w = current_w + (y(index, :).*S(index))./N ; %
end