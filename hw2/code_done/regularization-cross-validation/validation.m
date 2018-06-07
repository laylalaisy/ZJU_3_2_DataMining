%% Ridge Regression
%{
load('digit_train', 'X', 'y');

% Do feature normalization
[P, N] = size(X);
mean = zeros(P,1);    % mean
for i = 1:P
    mean(i,1) = sum(X(i, :)) / N;
end
variance = sqrt(var(X, 1, 2));    % variance
for i = 1:P     % normalization
    for j = 1:N
        if(abs(variance(i))>0.00001)
            X(i,j) = 1.0 * (X(i,j) - mean(i)) / variance(i);
        else
            X(i,j) = 0.0;
        end
    end
end

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
E_val_min = 10000000000000;

for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        % take point j out of X
        X_ = X; 
        X_(:, j) = [];
        y_ = y; 
        y_(:, j) = [];
        % get w
        w = ridge(X_, y_, lambdas(i));
        % single observation 
        if(sign(w' * [1; X(:,j)]) ~= y(j))
            E_val = E_val + 1;
        end
    end
    % Update lambda according validation error
    if E_val < E_val_min
        E_val_min = E_val;
        lambda = lambdas(i);
    end
end

fprintf('Lambda chosen by LOOCV is %f.\n', lambda);

% without regression
w_without = ridge(X, y, 1e-12);
fprintf('Without regularization, the sum of omega square is %f.\n', w_without'*w_without);

% with regression
w_with = ridge(X, y, lambda);
fprintf('With regularization, the sum of omega square is %f.\n', w_with'*w_with);

% Compute training error
[P, N] = size(X);
y_predict_without = sign((w_without') * [ones(1,N);X]);
E_train_without = sum(y_predict_without ~= y) * 1.0 / N;
fprintf('Without regularization, the train error is %f.\n', E_train_without);
y_predict_with = sign((w_with')*[ones(1,N);X]);
E_train_with = sum(y_predict_with ~= y) * 1.0/N;
fprintf('With regularization, the train error is %f.\n', E_train_with);

load('digit_test', 'X_test', 'y_test');
% Do feature normalization
[P, N] = size(X_test);
mean = zeros(P,1);    % mean
for i = 1:P
    mean(i,1) = sum(X_test(i, :)) / N;
end
variance = sqrt(var(X_test, 1, 2));    % variance
for i = 1:P     % normalization
    for j = 1:N
        if(abs(variance(i))>0.00001)
            X_test(i,j) = 1.0 * (X_test(i,j) - mean(i)) / variance(i);
        else
            X_test(i,j) = 0.0;
        end
    end
end

% Compute test error
[P, N] = size(X_test);
y_predict_without = sign((w_without') * [ones(1,N);X_test]);
E_test_without = sum(y_predict_without ~= y_test) * 1.0 / N;
fprintf('Without regularization, the test error is %f.\n', E_test_without);
y_predict_with = sign((w_with')*[ones(1,N);X_test]);
E_test_with = sum(y_predict_with ~= y_test) * 1.0/N;
fprintf('With regularization, the test error is %f.\n', E_test_with);
%}

%% Logistic
load('digit_train', 'X', 'y');

% Do feature normalization
[P, N] = size(X);
mean = zeros(P,1);    % mean
for i = 1:P
    mean(i,1) = sum(X(i, :)) / N;
end
variance = sqrt(var(X, 1, 2));    % variance
for i = 1:P     % normalization
    for j = 1:N
        if(abs(variance(i))>0.00001)
            X(i,j) = 1.0 * (X(i,j) - mean(i)) / variance(i);
        else
            X(i,j) = 0.0;
        end
    end
end

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
E_val_min = 10000000000000;

for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        % take point j out of X
        X_ = X; 
        X_(:, j) = [];
        y_ = y; 
        y_(:, j) = [];
        % get w
        w = logistic_r(X_, y_, lambdas(i));
        % single observation 
        if(sign(w' * [1; X(:,j)]) ~= y(j))
            E_val = E_val + 1;
        end
    end
    % Update lambda according validation error
    if E_val < E_val_min
        E_val_min = E_val;
        lambda = lambdas(i);
    end
end

fprintf('Lambda chosen by LOOCV is %f.\n', lambda);

% without regression
w_without = logistic_r(X, y, 1e-12);
fprintf('Without regularization, the sum of omega square is %f.\n', w_without'*w_without);

% with regression
w_with = logistic_r(X, y, lambda);
fprintf('With regularization, the sum of omega square is %f.\n', w_with'*w_with);

% Compute training error
[P, N] = size(X);
y_predict_without = sign((w_without') * [ones(1,N);X]);
E_train_without = sum(y_predict_without ~= y) * 1.0 / N;
fprintf('Without regularization, the train error is %f.\n', E_train_without);
y_predict_with = sign((w_with')*[ones(1,N);X]);
E_train_with = sum(y_predict_with ~= y) * 1.0/N;
fprintf('With regularization, the train error is %f.\n', E_train_with);

load('digit_test', 'X_test', 'y_test');
% Do feature normalization
[P, N] = size(X_test);
mean = zeros(P,1);    % mean
for i = 1:P
    mean(i,1) = sum(X_test(i, :)) / N;
end
variance = sqrt(var(X_test, 1, 2));    % variance
for i = 1:P     % normalization
    for j = 1:N
        if(abs(variance(i))>0.00001)
            X_test(i,j) = 1.0 * (X_test(i,j) - mean(i)) / variance(i);
        else
            X_test(i,j) = 0.0;
        end
    end
end

% Compute test error
[P, N] = size(X_test);
y_predict_without = sign((w_without') * [ones(1,N);X_test]);
E_test_without = sum(y_predict_without ~= y_test) * 1.0 / N;
fprintf('Without regularization, the test error is %f.\n', E_test_without);
y_predict_with = sign((w_with')*[ones(1,N);X_test]);
E_test_with = sum(y_predict_with ~= y_test) * 1.0/N;
fprintf('With regularization, the test error is %f.\n', E_test_with);

%% SVM with slack variable
