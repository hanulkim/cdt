function features = ExtractFeatures(img, samples)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2015, Hanul Kim
% 
% This file is part of the MOT code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
num_sample = size(samples, 1);
features = zeros(3520, num_sample);
for n = 1:num_sample
  xmin = samples(n,1); xmax = samples(n,1)+samples(n,3)-1;
  ymin = samples(n,2); ymax = samples(n,2)+samples(n,4)-1;
  img_ = single(imResample(img(ymin:ymax, xmin:xmax, :), [40, 40]));
    
  feat_color = colorMex(img_);
  feat_grad = fhog(img_, 4, 9, .2, 1);
  feat_grad = feat_grad(:,:,1:31);
  feat = cat(3, feat_color, feat_grad);
  feat = reshape(feat, [64,55]);
  features(:,n) = reshape(normr(feat), [3520,1]);
end
features = normc(features);

end