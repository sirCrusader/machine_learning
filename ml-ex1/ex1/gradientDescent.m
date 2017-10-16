function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    J_f = 0;
    J_s = 0;
    for index = 1:m
      J_f = J_f + (X(index,1) * theta(1) + X(index,2) * theta(2) - y(index));
      J_s = J_s + (X(index,1) * theta(1) + X(index,2) * theta(2) - y(index)) * X(index, 2);
    end

    J_f = J_f / m;
    J_s = J_s / m;
    
    temp(1) = theta(1) - alpha * J_f;
    temp(2) = theta(2) - alpha * J_s;
    theta = [temp(1); temp(2)];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
