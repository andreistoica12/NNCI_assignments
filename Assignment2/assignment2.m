% Legend:
% n_D = number of runs
% alphas = ratios of P/N (array)
% P_vals = numbers of data points (array)
% N_gen_errors = generalization errors for each data point and alpha (length(P_vals) x length(alphas))
% n_max = factor for determining a reasonable t_max
% t_max = n_max * P
% N = P / alpha
% y = P randomly-generated N-dimensional training samples (P x N)


n_D = 50; % number of runs

% alphas = [0.25, 0.5, 1, 3.0, 5.0];
alphas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]; % alphas as in assignment
% alphas = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0];

P_vals = [5, 20, 35];
% P_vals = [5, 20, 40, 60, 80, 100]; % change to P values ?
N_gen_errors = zeros(length(P_vals),length(alphas)); % should be for number of Ps
n_max = 125;

mean_t_max = zeros(length(P_vals), length(alphas));

for p_value = 1:length(P_vals)
    P = P_vals(p_value);
    t_max = n_max * P;  % number of epochs
    gen_errors = zeros(1, length(alphas));

%     % not sure if this makes sense if we have n_D runs
%     t_max_final = zeros(1, length(alphas)); 
%     t_max_final(:) = t_max;

    for x = 1:length(alphas)
        alpha = alphas(x);
        gen_error = 0;
        
        % for each run, we find the t_max and store it in the n-dimensional
        % array n_t_max
        n_t_max = zeros(1, n);
        n_t_max(:) = t_max;     % we initialize the n_t_max array with default t_max

        for n = 1:n_D
            N = int64(P/alpha); % dimension size
            
            % we create P randomly-generated N-dimensional vectors
            y = randn(P, N);

            % we define the teacher weight vector as a column vector
            w_star = randn(N, 1);
            % we normalise the teacher weight vector w* to 1, then multiply each element with
            % the preferred value sqrt(N) so that the squared norm is equal
            % to N
            w_star_norm = norm(w_star);
            w_star = w_star ./ w_star_norm .* sqrt(double(N));

%             disp(norm(w_star)^2);

            % we initialize the target values dependent on w* and y with
            % zeros
            S = zeros(P, 1);

            % we determine the target values with the help of the teacher
            % perceptron
            for item = 1:P
                temp = y(item, :) * w_star;
                if temp < 0
                    S(item) = -1;
                else
                    S(item) = 1;
                end 
            end 
            
            % initialise the weights at 0
            w = zeros(N, t_max);
            k = zeros(P, 1);
            E = zeros(P, 1);
            
            for epoch = 1:t_max
                for example = 1:P
                    % not okay here, current_w will always be 0 first, we
                    % should take the previous value at epoch - 1
                    current_w = w(:, epoch);
                    E(example) = (y(example, :) * current_w) * S(example);
                    if norm(current_w) ~= 0 % ensure no division by 0
                        k(example) = E(example) / norm(current_w);
                    else
                        % if the norm of the weight vector is 0, then all
                        % elements are 0, so E(example) should also be 0
                        k(example) = E(example);
                    end
                end
                % we determine the example with lowest stability
                [min_example, min_index] = min(k); % find(k == min(k))
                
                % weight vectors don't update
                % NEEDS FIXING
                w(:, epoch) = current_w + ((y(min_index,:)' .* S(min_index)) ./ double(N)) ;  % update the weights


                % TODO: determine a good stopping criterion
                % 2 vectors are near parallel if w1 * w2 / (|w1| * |w2|) -> 1. acos(1) = 0
                % So, if acos(w1*w2/(|w1|*|w2|)) / pi is close to 0, then
                % the 2 vectors are near parallel, so we found a version
                % space close to the teacher (a good rule)

                threshold = int64(t_max / 4);
                if epoch > threshold
                    % TODO: solve this argument (it seems to be always 1)
%                     fprintf("epoch = %d, epoch - threshold = %d \n", epoch, epoch-threshold);
%                     fprintf("threshold = %d \n", threshold);
                    argument = dot(w(:, 1), w(:, epoch)) / norm(w(:, 1)) / norm(w(:, epoch));
%                     fprintf("argument = %d \n", argument);
                    angular_change = acos(argument) / pi ;
                    fprintf("angular change = %d \n", angular_change);

                    if angular_change < 0.3
                        n_t_max(1, n) = epoch ;
                        break;
                    end
                end
            end
            t_max_run_n = n_t_max(1, n);
            gen_error = gen_error + acos(dot(w(:, t_max_run_n), w_star) / ...
                (norm(w(:, t_max_run_n)) * norm(w_star))) / pi ;
%             disp(w(:, t_max_run_n));
%             fprintf(" student above \n techer below \n");
%             disp(w_star);
        end
        % the vector of all generalization errors for each alpha
        % it is computed as the average of generalization errors over the
        % number of n_D randomized data sets per value of P
        gen_errors(1, x) = gen_error / n_D ;
        % we compute the mean t_max for all n_D runs with alpha = x and P =
        % P_vals(p_value)
        mean_t_max(p_value, x) = mean(n_t_max);
    end
    % the matrix of all generalization errors for each alpha, for each
    % chosen number of P input vectors
    N_gen_errors(p_value,:) = gen_errors ;
end

% TODO: plot
% plotting the results
figure;
plot(alphas, N_gen_errors(1, :), '-*');
title('$\epsilon_{g}$ as a function of $\alpha$', 'Interpreter', 'latex');
xlabel("$\alpha = P/N$",'Interpreter', 'latex');
ylabel("$\epsilon_{g}$", 'Interpreter', 'latex');
hold on;
for p = 2:length(P_vals)
    plot(alphas, N_gen_errors(p,:), '-*');
end
hold off;
legendCell = cellstr(num2str(P_vals', 'P=%-d'));
legend(legendCell);

% TODO: test 
% TODO: iris flower data (extension)
% TODO: write report :)
