function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

y = (y == 1);

[P, N] = size(X);       
X = [ones(1, N); X];    % add 1 dimension to X for w0
w = rand(P+1, 1);

iter = 0;
stop_iter = 1000;
stop_loss = 0;
learning_rate = 0.02;

h = 1 ./ (1 + exp(-1 * w' * X));    % sigmoid function: 1 * N
loss = -(1 / N) * (y * log(h)' + (1 - y) * log(1 - h)');    % loss value

while(loss >= stop_loss && iter <= stop_iter)
    gradient = X * (h - y)';       % gradient: P+1 * 1;
    w = w - learning_rate * gradient;   % update weight: 1 * N
    h = 1.0 ./ (1 + exp(-1.0 * w' * X));    % update h
    loss = -(1.0 / N) * (y * log(h)' + (1 - y) * log(1 - h)');    % update loss
    iter = iter + 1;
end


end
