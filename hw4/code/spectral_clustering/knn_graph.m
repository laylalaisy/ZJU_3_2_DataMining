function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE

[N, P] = size(X);
Dist = pdist2(X, X);
W = zeros(N, N);

for i = 1:N
    [sort_dist, sort_index] = sort(Dist(i, :),2);
    for j = 1:k
        if sort_dist(1, j) < threshold
            W(i, sort_index(1, j)) = 1;
        end
    end
end

end


