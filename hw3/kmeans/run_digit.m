load('digit_data.mat');

[n, p] = size(X);

K = [10, 20, 50];

for i = 1:length(K)
    k = K(i);
    [idx, ctrs, iter_ctrs] = kmeans(X, k);
    figure;
    show_digit(ctrs);
end