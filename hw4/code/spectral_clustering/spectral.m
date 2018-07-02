function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE
N = size(W, 1);

D = diag(sum(W, 2));
L = D - W;

% calculate eigenvalue
[eig_vector, eig_value] = eig(L);
% sort eigenvalue
[sort_eig_value_value, sort_eig_value_index] = sort(diag(eig_value));
eig_vector_k = eig_vector(:, sort_eig_value_index(1:k));

[idx, center, bCon, sumD, D] = litekmeans(eig_vector_k, k);

end