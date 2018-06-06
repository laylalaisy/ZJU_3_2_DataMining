function [w, iter] = perceptron(X, y)
% Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
[P, N] = size(X);       % get size of X
X = [ones(1, N); X];    % add 1 dimension to X for w0           
w = ones(P+1, 1);       % weight
d = zeros(1, N);        % judge: (w' * X(:, i)).* y(i)

iter = 0;               % train times: in case of infinity
ok = 0;                 % test if all (w' * X(:, i)).* y(i) > 0, means right predict label

while(ok == 0 && iter ~= 2000)
    iter = iter+1;
    for i=1:N
        d(i) = (w' * X(:, i)).* y(i);   % judge
        if  d(i) <= 0                   % wrong predict    
            w = w + X(:, i) .* y(i);    % adjust weight
            break;
        end
    end
    if d(:) > 0    % all predict correct, then end training
        ok = 1;
    end
end