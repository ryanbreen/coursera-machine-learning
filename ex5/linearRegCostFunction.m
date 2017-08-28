function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

reg = sum(theta(2:end).^2);

h_of_x = X * theta;
delta = h_of_x - y;

J = (1/(2*m)) * (sum(delta.^2) + (lambda * reg));
grad = (1/m) .* sum((h_of_x - y) .* X, 1) + (lambda / m) .* theta.';
grad(1) = (1/m) * sum((h_of_x(:, 1) - y(:, 1)) .* X(:, 1));

% =========================================================================

grad = grad(:);

end
