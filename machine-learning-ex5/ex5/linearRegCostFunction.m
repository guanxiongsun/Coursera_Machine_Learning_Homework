function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H_theta = X*theta;
J_1 = sum((H_theta - y).^2);
d = size(theta,1);
temp = theta(2:d,:).^2;
J_2 =  sum(temp);
J = (1/(2*m))*J_1 + (lambda/(2*m))*J_2;

for j = 1:size(theta)
  if (j==1)
    grad(j) = (1/m)* sum(H_theta - y);
  else
    grad(j) = (1/m)* sum((H_theta - y).*X(:,j)) + (lambda/m)*theta(j);
  endif
endfor

% =========================================================================

J = J;
grad = grad(:);

end
