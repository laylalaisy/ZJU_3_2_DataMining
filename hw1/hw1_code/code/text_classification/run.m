% ham_train: contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train: contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');

% N: is the size of vocabulary.
N = size(ham_train, 2);

% There are 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;

% Do smoothing
x = [ham_train;spam_train] + 1;

% ham_test: contains the occurences of each word in each ham test email. 
% P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;

% spam_test: contains the occurences of each word in each spam test email. 
% Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

% TODO
% Implement a ham/spam email classifier, and calculate the accuracy of your classifier

%% find top 10 spam words
L = likelihood(x);

word_spam = zeros(1, N);
for i=1:N
    word_spam(1,i) = L(2,i)/L(1,i);
end

% sorted ratio from min to max and store each index
[sorted, index] = sort(word_spam(1,:));

% list top 10 spam words' index
for i=N:-1:N-9
    disp(index(i));
end

%% spam accuracy
L_log = log(L);

Prior_ham = log(num_ham_train / (num_ham_train + num_spam_train));
Prior_spam = log(num_spam_train / (num_ham_train + num_spam_train));

% log(posterior)
[Q, N] = size(spam_test);    % Q is number of spam test emails
Class_spam = zeros(1,Q);     % ham=1, spam=2

Posterior_spam = spam_test*L_log';    % posterior: Q * 2
for i=1:Q
    Posterior_spam(i,1)=Posterior_spam(i,1) + Prior_ham;    % add log(prior probability)
    Posterior_spam(i,2)=Posterior_spam(i,2) + Prior_spam;
end   

% compare Posterior as ham and as spam, the larger one is the predict class
error_spam = 0;
for i=1:Q
    if Posterior_spam(i,1) > Posterior_spam(i,2)
        Class_spam(1,i) = 1;
        error_spam = error_spam+1;
    else
        Class_spam(1,i) = 2;
    end
end

TP = Q - error_spam;
FN = error_spam;

%% ham accuracy
% log(posterior)
[P, N] = size(ham_test);    % P is number of ham test emails
Class_ham = zeros(1,P);     % ham=1, spam=2

Posterior_ham = ham_test*L_log';    % posterior: P * 2
for i=1:P
    Posterior_ham(i,1)=Posterior_ham(i,1) + Prior_ham;    % add log(prior probability)
    Posterior_ham(i,2)=Posterior_ham(i,2) + Prior_spam;
end   

% compare Posterior as ham and as spam, the larger one is the predict class
error_ham = 0;
for i=1:P
    if Posterior_ham(i,1) > Posterior_ham(i,2)
        Class_ham(1,i) = 1;
    else
        Class_ham(1,i) = 2;
        error_ham = error_ham+1;
    end
end

TN = P - error_ham;
FP = error_ham;

%% precision & recall
accuracy_spam = (TP+TN) / (P+Q);
precision = TP/(TP+FP);
recall = TP/(TP+FN);

disp('accuracy of spam:');
disp(accuracy_spam);

disp('TP:');
disp(TP);
disp('FP');
disp(FP);
disp('FN');
disp(FN);
disp('TN');
disp(TN);
disp('precision:');
disp(precision);
disp('recall');
disp(recall);






