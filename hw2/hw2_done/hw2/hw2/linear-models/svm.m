function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE

[P, N] = size(X);
X = [ones(1, N); X];
w = quadprog(eye(P+1), [], -diag(y)*(X'), -ones(N,1));    % weight

d = y .* (w' * X);
num = sum(abs(d - 1) < 0.01);

end
