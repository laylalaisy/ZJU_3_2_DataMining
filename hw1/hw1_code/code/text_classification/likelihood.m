function l = likelihood(x)
% LIKELIHOOD Different Class Feature Liklihood 
%
%   INPUT:  x, features of different class, C-By-N vector
%           C is the number of classes, N is the number of different feature
%
%   OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N matrix
%

[C, N] = size(x);
l = zeros(C, N);

%TODO: P(x|wi)
total = zeros(1,C);
for i=1:C
    total(1, i)=sum(x(i,:));
end

for i=1:C
    for j=1:N
        l(i,j)=x(i,j)/total(1, i);
    end
end

end
