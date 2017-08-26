function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

function [a1, a2, a3] = feedForward(X_m)
    a1 = [1 X_m];

    a2 = sigmoid(a1 * Theta1');

    a2 = [1 a2];

    a3 = sigmoid(a2 * Theta2');
end

function [delta2, delta3] = backProp(a2, a3, y_t)
    delta3 = (a3 - y_t);
    delta2 = (delta3 * Theta2) .* a2 .* (1-a2);
end

cost = zeros(m, num_labels);

for i = 1:m
    
    % create a m x num_labels version of y
    y_expanded = zeros(1, num_labels);
    if y(i) == 0
        y_expanded(10) = 1;
    else
        y_expanded(y(i)) = 1;
    end
    
    [a1, a2, a3] = feedForward(X(i, :));
    cost(i, :) = -(y_expanded .* log(a3)) - ((1 - y_expanded) .* (log(1 - a3)));
    
    [delta2, delta3] = backProp(a2, a3, y_expanded);
    
    Delta1 = (delta2.'*a1);
    Delta1 = Delta1(2:end, :);
    Delta2 = (delta3.'*a2);
    
    Theta1_grad = Theta1_grad + Delta1;
    Theta2_grad = Theta2_grad + Delta2;
end

regularized_theta1 = Theta1(:, 2:end);
regularized_theta1 = regularized_theta1 .* regularized_theta1;

regularized_theta2 = Theta2(:, 2:end);
regularized_theta2 = regularized_theta2 .* regularized_theta2;

regular_term = (lambda / (2 * m)) * (sum(regularized_theta1(:)) + sum(regularized_theta2(:)));

J = (1/m) * sum(cost(:)) + regular_term;

reg_theta1 = (lambda / m) * Theta1;
reg_theta1(:, 1) = zeros(1, size(Theta1, 1));

reg_theta2 = (lambda / m) * Theta2;
reg_theta2(:, 1) = zeros(1, size(Theta2, 1));

Theta1_grad = Theta1_grad .* (1/m) + reg_theta1;
Theta2_grad = Theta2_grad .* (1/m) + reg_theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
