function [X, y] = mktestdata(N, w)
%MKDATA Generate data set.
%
%   INPUT:  N: number of samples.
%           w: target function parameters, (P+1)-by-1 column vector.
%
%   OUTPUT: X: sample features, P-by-N matrix.
%           y: sample labels, 1-by-N row vector.
%           


range = [-1, 1];
dim = 2;

X = rand(dim, N)*(range(2)-range(1)) + range(1); 
while true  
  y = sign(w'*[ones(1, N); X]);
  if all(y)
      break; 
  end
end

end