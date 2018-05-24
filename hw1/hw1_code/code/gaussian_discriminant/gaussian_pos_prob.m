function P = gaussian_pos_prob(X, Mu, Sigma, Phi)
%  GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);     % number of data points
K = length(Phi);    % number of class
P = zeros(N, K);    % posterior probability: N * K

% Your code HERE P(y=k|x)=P(x|y=k)*P(y=k)/P(x)
% P(x|y=k) Likelihood 

L = zeros(N, K); 
for i=1:N
    sum = 0;
    % P(x) y=1...k
    for j=1:K
        L(i,j)=1/(2*pi*sqrt(det(Sigma(:,:,j))))*exp(-1/2*(X(:,i)-Mu(:,j))'*inv(Sigma(:,:,j))*(X(:,i)-Mu(:,j)));
        sum = sum + L(i,j)*Phi(1,j);
    end
    % P(y=k|x)=P(x|y=k)*P(y=k)/P(x)
    for j=1:K
        P(i,j)=L(i,j)*Phi(1,j)/sum;
    end
end








