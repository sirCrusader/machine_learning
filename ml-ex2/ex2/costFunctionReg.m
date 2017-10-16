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

cost_1 = 0;
cost_2 = 0;

for index =1:m
  sum_result = sum(theta'.*X(index, :));
  z = sigmoid(sum_result);
  cost_1 = -1 * y(index) * log(z);
  cost_2 = (1 - y(index)) * log(1 - z);
  J = J + (cost_1 - cost_2);
end

theta_sum = 0;
for theta_index=2:length(theta)
  theta_sum = theta_sum + (theta(theta_index) ^ 2);
end

J = J / m + (lambda * theta_sum) / (2 * m);

temp = zeros(size(theta));

for index = 1:m
  sum_result = sum(theta'.*X(index, :));
  z = sigmoid(sum_result) - y(index);
  for s_index = 1:length(theta)
    temp(s_index) = temp(s_index) + z * X(index, s_index);
  end
end

temp(1) = temp(1) / m;
for s_index = 2:length(theta)
  temp(s_index) = temp(s_index) / m + (lambda * theta(s_index)) / m;
end

grad = temp;

% =============================================================

end
