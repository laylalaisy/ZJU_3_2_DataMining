% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron
nRep = 1000;     % number of replicates
nTrain = 100;     % number of training data
nTest = 100;     % number of test data

E_train = 0;     % training error rate
E_test = 0;      % test error rate
iter_sum = 0;    % sum of iter

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    [w_g, iter] = perceptron(X, y);
    
    % Compute training error
    X_train = [ones(1, nTrain); X];    % add 1 dimension to X for w0
    predict = w_g' * X_train;     % predict label
    for j = 1:nTrain
        if (predict(j) * y(j)) < 0     % predict wrong
            E_train = E_train + 1;
        end
    end
    % Compute testing error
    [X_test, y_test] = mktestdata(nTest, w_f);
    X_test = [ones(1, nTest); X_test];    % add 1 dimension to X for w0
    predict = w_g' * X_test;     % predict label
    for j = 1:nTest
        if (predict(j) * y_test(j)) < 0     % predict wrong
            E_test = E_test + 1;
        end
    end
    % Sum up number of iterations
    iter_sum = iter_sum + iter;
end

E_train = E_train / (nRep * nTrain);
E_test = E_test / (nRep * nTest);
avgIter = iter_sum / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X, y, w_f, w_g, 'Pecertron');

%% Part2: Preceptron: Non-linearly separable case
nTrain = 50; % number of training data
[X, y, w_f] = mkdata(nTrain, 'noisy');
[w_g, iter] = perceptron(X, y);
disp(iter);

%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 100; % number of test data

E_train = 0;     % training error rate
E_test = 0;      % test error rate

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    w_g = linear_regression(X, y);

    % Compute training error
    X_train = [ones(1, nTrain); X];    % add 1 dimension to X for w0
    predict = w_g' * X_train;     % predict label
    for j = 1:nTrain
        if (predict(j) * y(j)) < 0     % predict wrong
            E_train = E_train + 1;
        end
    end
    % Compute testing error
    [X_test, y_test] = mktestdata(nTest, w_f);
    X_test = [ones(1, nTest); X_test];    % add 1 dimension to X for w0
    predict = w_g' * X_test;     % predict label
    for j = 1:nTest
        if (predict(j) * y_test(j)) < 0     % predict wrong
            E_test = E_test + 1;
        end
    end
end

E_train = E_train / (nRep * nTrain);
E_test = E_test / (nRep * nTest);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 100;      % number of training data

E_train = 0;     % training error rate
E_test = 0;      % test error rate

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain, 'noisy');
    w_g = linear_regression(X, y);
    
    % Compute training error
    X_train = [ones(1, nTrain); X];    % add 1 dimension to X for w0
    predict = w_g' * X_train;     % predict label
    for j = 1:nTrain
        if (predict(j) * y(j)) < 0     % predict wrong
            E_train = E_train + 1;
        end
    end
    % Compute testing error
    [X_test, y_test] = mktestdata(nTest, w_f);
    X_test = [ones(1, nTest); X_test];    % add 1 dimension to X for w0
    predict = w_g' * X_test;     % predict label
    for j = 1:nTest
        if (predict(j) * y_test(j)) < 0     % predict wrong
            E_test = E_test + 1;
        end
    end
end

E_train = E_train / (nRep * nTrain);
E_test = E_test / (nRep * nTest);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');

[P_train, N_train] = size(X);       % get size of X
[P_test, N_test] = size(X_test);    % get size of X

w = linear_regression(X, y);

E_train = 0;     % training error rate
E_test = 0;      % test error rate

% Compute training error
X_train = [ones(1, N_train); X];          % add 1 dimension to X for w0
predict_train = w_g' * X_train;           % predict label
for j = 1:N_train
    if (predict_train(j) * y(j)) < 0      % predict wrong
        E_train = E_train + 1;
    end
end

% Compute testing error
X_test = [ones(1, N_test); X_test];             % add 1 dimension to X for w0
predict_test = w_g' * X_test;               % predict label
for j = 1:N_test
    if (predict_test(j) * y_test(j)) < 0    % predict wrong
        E_test = E_test + 1;
    end
end

E_train = E_train / N_train;
E_test = E_test / N_test;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');

[P_train, N_train] = size(X);       % get size of X
[P_test, N_test] = size(X_test);    % get size of X
% CHANGE THIS LINE TO DO TRANSFORMATION
X_t = [X; X(1,:).*X(2, :); X(1, :).^2; X(2, :).^2]; 

w_fit = linear_regression(X_t, y);

% CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t = [X_test; X_test(1,:).*X_test(2, :); X_test(1, :).^2; X_test(2, :).^2];

E_train = 0;     % training error rate
E_test = 0;      % test error rate

% Compute training error
X_t = [ones(1, N_train); X_t];    % add 1 dimension to X for w0
predict_train = w_fit' * X_t;         % predict label
for j = 1:N_train
    if (predict_train(j) * y(j)) < 0      % predict wrong
        E_train = E_train + 1;
    end
end

% Compute testing error
X_test_t = [ones(1, N_test); X_test_t];    % add 1 dimension to X for w0
predict_test = w_fit' * X_test_t;              % predict label
for j = 1:N_test
    if (predict_test(j) * y_test(j)) < 0   % predict wrong
        E_test = E_test + 1;
    end
end

E_train = E_train / N_train;
E_test = E_test / N_test;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 100; % number of test data

E_train = 0;     % training error rate
E_test = 0;      % test error rat

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    w_g = logistic(X, y);
    % compute training error
    E_train = E_train + sum(sign(w_g' * [ones(1,nTrain); X]) ~= y);
  
    [X_test, y_test] = mktestdata(nTest, w_f);
    % compute training error
    E_test = E_test + sum(sign(w_g' * [ones(1,nTest); X_test]) ~= y_test);
end

E_train = E_train / (nRep * nTrain);
E_test = E_test / (nRep * nTest);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10000; % number of training data

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain, 'noisy');
    w_g = logistic(X, y);
    % compute training error
    E_train = E_train + sum(sign(w_g' * [ones(1,nTrain); X]) ~= y);
  
    [X_test, y_test] = mktestdata(nTest, w_f);
    % compute training error
    E_test = E_test + sum(sign(w_g' * [ones(1,nTest); X_test]) ~= y_test);
end

E_train = E_train / (nRep * nTrain);
E_test = E_test / (nRep * nTest);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression: noisy');

%% Part8: SVM
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 100; % number of test data

number = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    [w_g, num_sc] = svm(X, y);
    % Compute training error
    E_train = E_train + sum(y .* (w_g' * [ones(1,nTrain); X]) < 1);
    
    [X_test, y_test] = mktestdata(nTest, w_f);
    % Compute testing error
    E_test = E_test + sum(y .* (w_g' * [ones(1,nTrain); X]) < 1);
    % Sum up number of support vectors
    number = number + num_sc;
end

E_train = E_train / (nRep * nTrain);
E_test = E_test / (nRep * nTest);

number = number / nRep;

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of support vectors is %f.\n', number);
plotdata(X, y, w_f, w_g, 'SVM');
