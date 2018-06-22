load('kmeans_data');

K = 2;
repeat = 1000;
[n, p] = size(X);

max_SD = 0;
max_idx = [];
max_ctrs = [];
max_iter_ctrs = [];

min_SD = 100000;
min_idx = [];
min_ctrs = [];
min_iter_ctrs = [];

for i = 1:repeat
    [idx, ctrs, iter_ctrs] = kmeans(X, K);
    
    SD = 0.0;
    for j = 1:n
        SD = SD+norm(X(j,:) - ctrs(idx(j),:));
    end
    
    if(SD > max_SD)
        max_SD = SD;
        max_idx = idx;
        max_ctrs = ctrs;
        max_iter_ctrs = iter_ctrs;
    end
    
    if(SD < min_SD)
        min_SD = SD;
        min_idx = idx;
        min_ctrs = ctrs;
        min_iter_ctrs = iter_ctrs;
    end
end

figure;
kmeans_plot(X,max_idx,max_ctrs,max_iter_ctrs);
figure;
kmeans_plot(X,min_idx,min_ctrs,min_iter_ctrs);