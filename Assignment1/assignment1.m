n_D = 50;
alphas = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0];
success_rates = zeros(length(alphas), 1);
for x = 1:length(alphas)
    alpha = 0.75; % alphas(x);
    succesfull_runs = 0;
    for n = 1:n_D   
        epochs = 100; % number of epochs
        N = 10; % dimension of feature vectors - use 20 and 40 at least
        P = cast(N*alpha, 'uint8'); % number of sets
        
        % we create P randomly-generated N-dimensional vectors
        y = randn(P, N);

        % we generate P random numbers with values either -1 or 1
        S = 2*(rand(1,P)>0.5) - 1;

        % intialize weights at 0
        w = zeros(N, 1);
        E = zeros(P, 1);

        % double for-loop for the sequential perceptron training
        for epoch = 1:epochs
            for example = 1:P % should probs rename this
                current_w = w;
                E(example) = dot(current_w,(y(example,:).*S(example)));
                if E <= 0
                    w = current_w + (y(example,:).*S(example))'/N ;
                end % dont need an else because the value doesn't change
            end
            %fprintf("E = %d\n", E);
            if all(E>0)
                %fprintf("success\n");
                succesfull_runs = succesfull_runs + 1; 
                break;
            end
        end
    end
    success_rates(x) = succesfull_runs / n_D ;
    fprintf("rate: %d\n", success_rates(x));
end 
