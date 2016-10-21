function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h_theta = zeros(m,1);

for i=1 : m
  temp = (theta')*(X(i,:)');
  h_theta(i) = sigmoid(temp);
end

sum = 0;
sum_2 = 0;
for i=1:m
  sum = sum+((-y(i)*log(h_theta(i))- (1- y(i))*log(1- h_theta(i))));
end

n = size(theta)(1,1);

for j=2:n
  sum_2 = sum_2+ theta(j)^2;
end

J = sum/m + (lambda*sum_2)/(2*m);

sum = 0;
for i =1: m
  sum = sum + (h_theta(i)-y(i))*X(i,1);
end

grad(1) = sum/m;

for j =2: size(theta)(1,1)
  sum = 0;
  for i =1: m
    sum = sum + (h_theta(i)-y(i))*X(i,j);
  end
  grad(j)= sum/m + (lambda*theta(j)/m);
 end



% =============================================================

end
