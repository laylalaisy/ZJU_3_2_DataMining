img = imread('sample.jpg');
fea = double(reshape(img, size(img, 1)*size(img, 2), 3));

% YOUR (TWO LINE) CODE HERE
K = [64];
for i = 1:length(K)
    k = K(i);
    [idx, ctrs, iter_ctrs] = kmeans(fea, k);
    fea_new = ctrs(idx, :);
    figure
    imshow(uint8(reshape(fea_new, size(img))));
end