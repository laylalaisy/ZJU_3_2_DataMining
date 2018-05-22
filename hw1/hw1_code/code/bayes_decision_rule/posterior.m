function p = posterior(x)
%POSTERIOR Two Class Posterior Using Bayes Formula
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
%

[C, N] = size(x);
l = likelihood(x);
total = sum(sum(x));

%TODO: P(wi|x) = P(x|wi)*P(wi)/P(x)
p = zeros(C, N);

% P(w1) & P(w2)
s1 = sum(x(1,:), 2);        % add each row: w1, number of w1             
s2 = sum(x(2,:), 2);        % add each row: w2, number of w2
p_w1 = s1/total;            % P(w1)
p_w2 = s2/total;            % P(w2)

% P(x)
p_x = zeros(1, N);          % get P(x)
for i=1:N
    p_x(1, i)=(x(1,i)+x(2,i))/total;
end

% P(wi|x)
for i=1:N
    p(1,i)=l(1,i)*p_w1/p_x(1,i);
    p(2,i)=l(2,i)*p_w2/p_x(1,i);
end

end
