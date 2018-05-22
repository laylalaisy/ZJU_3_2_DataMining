function l = likelihood(x)
%LIKELIHOOD Different Class Feature Liklihood 
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N matrix
%

[C, N] = size(x);
l = zeros(C, N);

%TODO: P(x|wi)
s1 = sum(x(1,:), 2);        % add each row: w1             
s2 = sum(x(2,:), 2);        % add each row: w2

for i=1:N
    l(1,i)=x(1,i)/s1;
    l(2,i)=x(2,i)/s2;
end

end
