function [theta, J_history_multi] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
J_history_multi = zeros(num_iters, 1);

for iter = 1:num_iters

    new_theta = zeros(n, 1);

    applied = theta.' .* X;
    deltas = sum(applied, 2) - y;
    
    for j = 1:n
        new_theta(j) = (1/m) * sum(deltas .* X(:, j));
    end
    
    theta = theta - alpha .* new_theta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history_multi(iter) = computeCostMulti(X, y, theta);

end

end
