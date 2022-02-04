n_D = 50; % number of runs

% alphas = [0.25, 0.5, 1, 3.0, 5.0];
alphas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]; % alphas as in assignment
% alphas = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0];

P_vals = [5, 20, 35];
% P_vals = [5, 20, 40, 60, 80, 100]; % change to P values ?
N_gen_errors = zeros(length(P_vals),length(alphas)); % should be for number of Ps
n_max = 100;

for p_value = 1:length(P_vals)
    P = P_vals(p_value);
    t_max = n_max * P;  % number of epochs
    gen_errors = zeros(1, length(alphas));
    t_max_final = zeros(1, length(alphas));
    t_max_final(:) = t_max;
    for x = 1:length(alphas)
        alpha = alphas(x);
        gen_error = 0;
        for n = 1:n_D
            N = int64(P/alpha); % dimension size
            
            % we create P randomly-generated N-dimensional vectors
            y = randn(P, N);

            % we define the weight vector as a column vector
            w_star = randn(N, 1);
            % we normalise the teacher weight vector w* to 1, then multiply each element with
            % the preferred value sqrt(N) so that the squared norm is equal
            % to N
            w_star_norm = norm(w_star);
            w_star = w_star ./ w_star_norm .* sqrt(double(N));

            % we initialize the target values dependent on w* and y with 0s
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
            
            for epoch = 0:t_max - 1
                for example = 1:P
                    if epoch == 0
                        current_w = zeros(N, 1);
                    else
                        current_w = w(:, epoch);
                    end
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
                w(:, epoch + 1) = current_w + (y(example,:)' * S(example)) ./ double(N) ;  % update the weights
                
                % TODO: determine a good stopping criterion
                if epoch > P
                    % TODO: solve this argument (it seems to be always 1)
                    argument = dot(w(:, epoch - P), w(:, epoch)) / ...
                        (norm(w(:, epoch - P)) * norm(w(:, epoch)));
%                     fprintf("argument = %d", argument);
                    angular_change = acos(argument) / pi ;
                    fprintf("angular change = %d \n", angular_change);


                    if angular_change < 0.1
                        t_max_final(1, x) = epoch ;
                        break;
                    end
                end
            end
            gen_error = gen_error + acos(dot(w(:, t_max), w_star) / ...
                (norm(w(:, t_max)) * norm(w_star))) / pi ;
        end
        gen_errors(1, x) = gen_error / n_D ; 
    end
    N_gen_errors(p_value,:) = gen_errors ;
end

% TODO: plot
% plotting the results
figure;
plot(alphas, mean(N_gen_errors), '-*');
title('$\epsilon_{g}$ as a function of $\alpha$', 'Interpreter', 'latex');
xlabel("$\alpha = P/N$",'Interpreter', 'latex');
ylabel("$\epsilon_{g}$", 'Interpreter', 'latex');
% hold on;
% for p = 2:size(N_gen_errors, 1)
%     plot(alphas, N_gen_errors(p,:), '-*');
% end
% hold off;
legendCell = cellstr(num2str(mean(N_gen_errors)', 'N=%-d'));
legend(legendCell);

% TODO: test 
% TODO: iris flower data (extension)
% TODO: write report :)
