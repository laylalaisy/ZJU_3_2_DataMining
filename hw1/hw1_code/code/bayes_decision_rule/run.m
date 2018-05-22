% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');                                           % read in data
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);   % 1x1500
range = [min(all_x), max(all_x)];                       % [-6,12]
train_x = get_x_distribution(x1_train, x2_train, range);% get histogram
test_x = get_x_distribution(x1_test, x2_test, range);   % get histogram

%% Part1 likelihood:
% get likelihood
l = likelihood(train_x);        % get likelihood

% draw histogram
bar(range(1):range(2), l');     % get matrix for histogram 
xlabel('x');                    % xlabel
ylabel('P(x|\omega)');          % ylabel
axis([range(1) - 1, range(2) + 1, 0, 0.5]); % axis's range

% TODO
% compute the number of all the misclassified x using maximum likelihood decision rule
% get predict matrix
[C, N] = size(l);
l_predict= zeros(1, N);     % 1 * range of all posible x   

for i=1:N                     
    if l(1,i)>l(2,i)
        l_predict(1,i)=1;    % if P(x|w1)>P(x|w2), choose w1
    else
        l_predict(1,i)=2;    % if P(x|w1)<P(x|w2), choose w2
    end
end

[C1, N1] = size(x1_test);
[C2, N2] = size(x2_test);

l_test_error_w1 = 0;    % should be w2 while is w1 now
l_test_error_w2 = 0;    % should be w1 while is w2 now

for i=1:N1                     
    if l_predict(1, x1_test(i)-range(1)+1) ~= 1   % adjust location by -range(1)+1 as drawing distribution
        l_test_error_w1=l_test_error_w1+1;              % according to maximum likelihood decision rule, should be w2 while w1
    end
end

for i=1:N2                     
    if l_predict(1, x2_test(i)-range(1)+1) ~= 2   % adjust location by -range(1)+1 as drawing distribution
        l_test_error_w2=l_test_error_w2+1;              % according to maximum likelihood decision rule, should be w1 while w2
    end
end

disp("likelihood test error:");
disp(l_test_error_w1+l_test_error_w2);

%% Part2 posterior:
% get posterior
p = posterior(train_x);

% draw histogram
bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
[C, N] = size(p);
p_predict= zeros(1, N);     % 1 * range of all posible x   

for i=1:N                     
    if p(1,i)>p(2,i)
        p_predict(1,i)=1;    % if P(w1|x)>P(w2|x), choose w1
    else
        p_predict(1,i)=2;    % if P(w1|x)<P(w2|x), choose w2
    end
end

[C1, N1] = size(x1_test);    % should be w2 while is w1 now
[C2, N2] = size(x2_test);    % should be w1 while is w2 now

p_test_error_w1 = 0;
p_test_error_w2 = 0;

for i=1:N1                     
    if p_predict(1, x1_test(i)-range(1)+1) ~= 1   % adjust location by -range(1)+1 as drawing distribution
        p_test_error_w1=p_test_error_w1+1;              % according to optimal bayes decision rule, should be w2 while w1
    end
end

for i=1:N2                     
    if p_predict(1, x2_test(i)-range(1)+1) ~= 2   % adjust location by -range(1)+1 as drawing distribution
        p_test_error_w2=p_test_error_w2+1;              % according to optimal bayes decision rule, should be w1 while w2
    end
end

disp("beyes test error:");
disp(p_test_error_w1 + p_test_error_w2);

%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
bayes_error = [p_test_error_w2; p_test_error_w1];
total_risk = sum(risk * bayes_error);

disp("minimal total risk:");
disp(total_risk);
