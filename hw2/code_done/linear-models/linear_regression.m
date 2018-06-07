function w = linear_regression(X, y)
%LINEAR_REGRESSION Linear Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P, N] = size(X);       
X = [ones(1, N); X];    % add 1 dimension to X for w0

w = pinv(X * X') * X * y';
end
