function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%

% YOUR CODE HERE
[P, N_test] = size(X);
[P, N] = size(X_train);
y = zeros(1, N_test);

num_label = length(unique(y_train));
vote = zeros(num_label, 1);

dist = EuDist2(X', X_train');    % N_test * N
for i = 1:N_test
    vote = zeros(1, num_label);
    [dist_sort_value, dist_sort_index] = sort(dist(i, :), 'ascend');
    for j = 1:K
        cur_label = y_train(1, dist_sort_index(1, j));
        vote(1, cur_label + 1) = vote(1, cur_label + 1) + 1;
    end
    [max_vote_value, max_vote_index] = max(vote(1, :));
    y(1, i) = max_vote_index - 1;
end

end

