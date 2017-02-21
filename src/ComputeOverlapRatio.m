function overlap = ComputeOverlapRatio(ref, boxes)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2015, Hanul Kim
% 
% This file is part of the MOT code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

xmin = max(ref(1,1), boxes(:,1));
xmax = min(ref(1,1)+ref(1,3), boxes(:,1)+boxes(:,3));
ymin = max(ref(1,2), boxes(:,2));
ymax = min(ref(1,2)+ref(1,4), boxes(:,2)+boxes(:,4));

area1 = ref(1,3) * ref(1,4);
area2 = boxes(:,3) .* boxes(:,4);
i_area = max(0,(xmax-xmin)) .* max(0,(ymax-ymin));
u_area = area1 + area2 - i_area;
overlap = i_area ./ u_area;

end