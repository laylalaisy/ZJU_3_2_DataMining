load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');

% YOUR CODE HERE

% 1. Feature preprocessing
% 2. Run PCA
% 3. Visualize eigenface
% 4. Project data on to low dimensional space
% 5. Run KNN in low dimensional space
% 6. Recover face images form low dimensional space, visualize them

[N_train, P] = size(fea_Train);
[N_test, P] = size(fea_Test);

% show img
show_face(fea_Train);

dim_arr = [8, 16, 32, 64, 128];
for i = 1:length(dim_arr)
    % pca
    [eigvector_Train, eigvalue_Train] = pca(fea_Train, dim_arr(i));
    [eigvector_Test, eigvalue_Test] = pca(fea_Test, dim_arr(i));
    
    % reduce
    fea_Train_low = eigvector_Train' * fea_Train';
    fea_Test_low = eigvector_Test' * fea_Test';
    
    %knn
    y_Test = knn(fea_Train_low, fea_Test_low, gnd_Train',1)';
    
    % error rate     
    acc_rate = sum(gnd_Test' == y_Test') / N_test;
    fprintf('dimension:%d acc rate: %f, error rate: %f\n', dim_arr(i), acc_rate, 1-acc_rate);
    
    % show img
    figure;
    show_face(fea_Train_low' * );
end


    