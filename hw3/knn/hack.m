function digits = hack(img_name)
%HACK Recognize a CAPTCHA image
%   Inputs:
%       img_name: filename of image
%   Outputs:
%       digits: 1x5 matrix, 5 digits in the input CAPTCHA image.

load('hack_data');
% YOUR CODE HERE
cur_img_data = extract_image(img_name);
show_image(cur_img_data);

K = 5;
digits = knn(double(cur_img_data), double(X_train), double(y_train), K);
fprintf('The digits in the captcha is %d %d %d %d %d\n', digits);

end