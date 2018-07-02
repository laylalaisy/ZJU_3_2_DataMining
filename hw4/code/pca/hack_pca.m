function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation

img_r = double(imread(filename));

% YOUR CODE HERE
% show original img
figure;
imshow(uint8(img_r));   

% find pixel which is not backgound 
[img_row, img_col] = find(img_r < 255); 

% pca
[eigvector, eigvalue] = pca([img_row, img_col], 2);

angle = atand(eigvector(1) / eigvector(2));
img = imrotate(img_r, angle);

% show img after rotate
figure;
imshow(uint8(img))
end