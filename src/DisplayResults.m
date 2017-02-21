function [] = DisplayResults(img, targets, t)
% AUTORIGHTS
% -----------------------------------------------------------------------------
% Copyright (c) 2016, Hanul Kim
% 
% This file is part of the MOT code and is available under the terms of
% the Simplified BSD License provided in LICENSE. Please retain this notice 
% and LICENSE if you use this file (or any portion of it) in your project.
% -----------------------------------------------------------------------------

num_targets = numel(targets);
for i = 1:num_targets
  tt = t-targets(i).start+1;  
  img = insertShape(img,...
                    'Rectangle', targets(i).boxes(tt,:),...
                    'LineWidth', 4,...
                    'Color', [255, 0, 0]);        
end  
figure(1)
img_ = imresize(img, [480, 640]);
imshow(img_);
drawnow;

end