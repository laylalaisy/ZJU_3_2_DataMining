function [eigvector, eigvalue] = PCA(data, dim)
%PCA	Principal Component Analysis
%
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%

% YOUR CODE HERE
[N, P] = size(data);

minusmean = mean(data);    % mean of each feature
data_minusmean = data - repmat(minusmean, N, 1);

S = data_minusmean' * data_minusmean;
[eigvector, eigvalue] = eigs(S, dim, 'LM');  % max 2 eigvalue
eigvalue = diag(eigvalue)';

end