function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
sigmoid_x = sigmoid(X*theta);
log_sigmoid_x = log(sigmoid_x);
log_sigmoid_x_conj = log(1 - sigmoid_x);
theta_squared = theta.^2
theta_0 = theta(1)

J = sum((-1 / m)*(y'*log_sigmoid_x + (1-y')*log_sigmoid_x_conj)) + (lambda/(2*m))*(sum(theta_squared) - theta_0^2)
grad = (1/m)*X'*(sigmoid(X*theta)-y) + (lambda/m)*theta
grad(1) = grad(1) - (lambda/m)*theta_0


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
