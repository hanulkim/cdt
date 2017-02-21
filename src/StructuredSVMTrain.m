function model = StructuredSVMTrain(model, data, label)
% AUTORIGHTS
% -----------------------------------------------------------------------------
% Copyright (c) 2016, Hanul Kim
% 
% This file is part of the MOT code and is available under the terms of
% the Simplified BSD License provided in LICENSE. Please retain this notice 
% and LICENSE if you use this file (or any portion of it) in your project.
% -----------------------------------------------------------------------------

% -----------------------------------------------------------------------------
% Learning Parameters
% -----------------------------------------------------------------------------
data_dim = size(data,1);
num_data = size(data,2);
if isempty(model)
  max_epoch = 100;
  learn_rates = logspace(-1, -3, max_epoch);
  model = zeros(data_dim, 1);
else
  max_epoch = 20;
  learn_rates = logspace(-2, -3, max_epoch);    
end
alpha = 0.9;
lambda = 0.001;
batch_sz = 8;
momentum = zeros(data_dim, 1);

% -----------------------------------------------------------------------------
% Set Structured Data
% -----------------------------------------------------------------------------
diff_data = repmat(data(:,1), [1,num_data]) - data;
overlap_loss = ComputeOverlapLoss(label(1,:),label);

% -----------------------------------------------------------------------------
% Train model using SGD method with momentum
% -----------------------------------------------------------------------------
for epoch = 1:max_epoch
  random_idx = randperm(num_data);
  learn_rate = learn_rates(epoch);      
  for k = 1:batch_sz:num_data
    i_start = k;
    i_end = min(num_data, k+batch_sz-1);    
    idx_batch = random_idx(i_start:i_end);
    diff_batch = diff_data(:,idx_batch);
    loss_batch = overlap_loss(idx_batch);
    score_batch = model'*diff_batch;
    idx = logical(score_batch < loss_batch);
    gradient = lambda * model - sum(diff_batch(:,idx),2);  
    momentum = alpha * momentum - learn_rate * gradient; 
    model = model + momentum;
  end  
end

% for epoch = 1:max_epoch
%   random_idx = randperm(num_data);
%   learn_rate = learn_rates(epoch);      
%   for k = 1:num_data
%     i = random_idx(k);
%     if model'*diff_data(:,i) < overlap_loss(i)
%       gradient = lambda * model - diff_data(:,i);
%     else
%       gradient = lambda * model;
%     end    
%     momentum = alpha * momentum - learn_rate * gradient; 
%     model = model + momentum;
%   end  
% end

end
