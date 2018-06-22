function [out_sensitivity] = relu_backprop(in_sensitivity, in)
%The backpropagation process of relu
%   input paramter:
%       in_sensitivity  : the sensitivity from the upper layer, shape: 
%                       : [number of images, number of outputs in feedforward]
%       in              : the input in feedforward process, shape: same as in_sensitivity
%   
%   output paramter:
%       out_sensitivity : the sensitivity to the lower layer, shape: same as in_sensitivity

% TODO
[num_img, num_output] = size(in_sensitivity);
out_sensitivity = zeros(num_img, num_output);

for i = 1:num_img
    for j = 1:num_output
        if(in(i, j)>0)
            out_sensitivity(i, j) = in_sensitivity(i, j);
        end
    end
end

end



