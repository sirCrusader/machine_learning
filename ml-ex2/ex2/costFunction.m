function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
cost_1 = 0;
cost_2 = 0;

for index =1:m
  z = 1 / (1 + e ^ (-1 * (theta(1) * X(index, 1) + theta(2) * X(index, 2) + theta(3) * X(index, 3))));
  cost_1 = -1 * y(index) * log(z);
  cost_2 = (1 - y(index)) * log(1 - z);
  J = J + (cost_1 - cost_2);
end

J = J / m;

temp = zeros(size(theta));

for index = 1:m
  z = 1 / (1 + e ^ (-1 * (theta(1) * X(index, 1) + theta(2) * X(index, 2) + theta(3) * X(index, 3)))) - y(index);
  temp(1) = temp(1) + z * X(index, 1);
  temp(2) = temp(2) + z * X(index, 2);
  temp(3) = temp(3) + z * X(index, 3);
end
temp(1) = temp(1) / m;
temp(2) = temp(2) / m;
temp(3) = temp(3) / m;

grad = temp;

% =============================================================

end
