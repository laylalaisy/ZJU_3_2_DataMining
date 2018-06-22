%HACK_DATA save CAPTCHA images and labels
%   X_train: training imgs, P-by-N matrix
%   y_train: training labels, 1-by-N vector

img = dir("captcha/*.jpg");
num_img = length(img);

disp(img);

X_train = [];
y_train = [];

for i = 1:num_img
    cur_img_name = img(i).name;
    cur_img_path = strcat('./captcha/', cur_img_name);
    cur_img_data = extract_image(cur_img_path);
    X_train = [X_train, cur_img_data];
    for j = 1:5
        y_train = [y_train, str2num(cur_img_name(j))];
    end
end

save('hack_data.mat', 'X_train', 'y_train');

