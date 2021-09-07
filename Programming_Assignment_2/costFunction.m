function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
sigmoid_x = sigmoid(X*theta);
log_sigmoid_x = log(sigmoid_x);
log_sigmoid_x_conj = log(1 - sigmoid_x);
J = sum((-1 / m)*(y'*log_sigmoid_x + (1-y')*log_sigmoid_x_conj));
grad = (1/m)*X'*(sigmoid(X*theta)-y);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
