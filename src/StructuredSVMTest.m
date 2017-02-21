function score = StructuredSVMTest(model, data)
% AUTORIGHTS
% -----------------------------------------------------------------------------
% Copyright (c) 2016, Hanul Kim
% 
% This file is part of the MOT code and is available under the terms of
% the Simplified BSD License provided in LICENSE. Please retain this notice 
% and LICENSE if you use this file (or any portion of it) in your project.
% -----------------------------------------------------------------------------

score = model'*data;

end