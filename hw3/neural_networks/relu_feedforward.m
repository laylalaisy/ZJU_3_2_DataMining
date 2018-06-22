function [ out ] = relu_feedforward( in )
%The feedward process of relu
%   inputs:
%           in	: the input, shape: any shape of matrix
%   
%   outputs:
%           out : the output, shape: same as in

% TODO
% formula: ReLU(x) = max(0, X)
% e.g.: in: 400 * 25
out = in;
[num_img, num_input] = size(out);
for i = 1:num_img
    for j = 1:num_input
        if(out(i, j) < 0)
            out(i, j) = 0;
        end
    end
end

end
