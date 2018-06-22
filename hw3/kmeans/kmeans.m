function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE
[n, p] = size(X);

ctrs = X(randsample(n,K), :);    % K-by-p
last_ctrs = ctrs + 1;            % K-by-p

iter = 1;
threshold = 0.01;

while(norm(ctrs - last_ctrs) > threshold)
    iter_ctrs(:, :, iter) = ctrs;
    last_ctrs = ctrs;
    
    dist = EuDist2(X, ctrs);    % n-by-k

    idx = zeros(n, 1);
    % find cluster label
    for i = 1:n
        [min_dist, idx(i, 1)] = min(dist(i, :));
    end
    
    % collect each cluster
    for i = 1:K
        cluster_index = find(idx == i);    % index of the sample in current cluster
        ctrs(i,:) = mean(X(cluster_index, :));
    end
    
    iter = iter + 1;
end


end