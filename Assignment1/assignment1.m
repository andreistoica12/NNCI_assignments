n_D = 50; % change this to higher values for smoother 
% bonus alphas
alphas = [1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5];
% normal alphas
%alphas = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0];
%alphas = [0.70, 0.80, 0.90, 1.0, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.0, 2.10, 2.20, 2.30, 2.40, 2.5, 2.60, 2.70, 2.80, 2.90, 3.0];
epochs = 100; % number of epochs
Ns = [20, 40, 60, 80, 100]; % array of dimension of feature vectors - use 20 and 40 at least
N_success = zeros(length(Ns),length(alphas));
for n_value = 1:length(Ns)
    N = Ns(n_value);
    success_rates = zeros(length(alphas), 1);
    for x = 1:length(alphas)
        alpha = alphas(x);
        succesfull_runs = 0;
        for n = 1:n_D   
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
                    if E(example) <= 0
                        w = current_w + (y(example,:).*S(example))'/N ;
                    end % dont need an else because the value doesn't change
                end
                if all((E>0))
                    succesfull_runs = succesfull_runs + 1; 
                    break;
                end
            end
        end
        success_rates(x) = succesfull_runs / n_D ;
    end
    N_success(N,:) = success_rates;
end

% plotting the results
figure;
plot(alphas, N_success(1,:));
title('$Q_{l.s.}$ as a function of $\alpha$', 'Interpreter', 'latex');
xlabel("$\alpha = P/N$",'Interpreter', 'latex');
ylabel("$Q_{l.s.}$", 'Interpreter', 'latex');
hold on;
for p = 2:length(Ns)
    plot(alphas, N_success(p,:));
end
hold off;
lengendCell = cellstr(num2str(Ns', 'N=%-d'));
legend(lengendCell);
