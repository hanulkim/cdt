function trajectories = Tracking(conf, info, net, dets)
% AUTORIGHTS
% -----------------------------------------------------------------------------
% Copyright (c) 2016, Hanul Kim
% 
% This file is part of the MOT code and is available under the terms of
% the Simplified BSD License provided in LICENSE. Please retain this notice 
% and LICENSE if you use this file (or any portion of it) in your project.
% -----------------------------------------------------------------------------

fw_res_path = fullfile(conf.res_path, sprintf('%s_fw.mat',info.name));
if 0%exist(fw_res_path,'file')
  targets = importdata(fw_res_path);
else
  targets = ForwardPath(conf, info, net, dets);
  %save(fw_res_path,'targets','-v7.3');
end

bw_res_path = fullfile(conf.res_path, sprintf('%s_bw.mat',info.name));
if 0%exist(bw_res_path,'file')
  targets = importdata(bw_res_path);
else
  targets = BackwardPath(conf, info, net, targets); 
  %save(bw_res_path,'targets','-v7.3');
end
             
trajectories = CreateTrajectories(info, targets);                     

end

% -----------------------------------------------------------------------------
function targets = ForwardPath(conf, info, net, dets)
% -----------------------------------------------------------------------------
max_targets = 1000;
next_id = 1;
targets = struct('id',[],'start',[],'end',[],'classifier',[],'model',[],'boxes',[],'scores',[]);
targets(max_targets).id = [];

for t = 1:numel(info.im_list)  
  fprintf('[ForwardPath] frame: %d / %d\n',t,numel(info.im_list));  
  im = imread(fullfile(info.im_path, info.im_list{t}));
  [im_h,im_w,~] = size(im);
  
  % ---------------------------------------------------------
  % Tracking
  % ---------------------------------------------------------
  idx_targets = find([targets.end] == t-1);
  num_targets = numel(idx_targets);
  if num_targets > 0
    boxes = zeros(num_targets, 4);    
    features = zeros(3520, num_targets);
    for i = 1:num_targets    
      idx = idx_targets(i);  
      [boxes(i,:),features(:,i)] = ...
        ForwardTracking(im, targets(idx).boxes(end,:), targets(idx).model);    
    end 
           
    [~, scores] = GetDetectionScore(conf, im, net, boxes);
    states_detection = logical(scores > conf.lo_th)';
    states_occlusion = CheckOcclusion(conf, boxes, scores);
    states = states_detection & states_occlusion;
    for i = 1:num_targets 
      idx = idx_targets(i);
      if states(i) == 1
        targets(idx).boxes = cat(1, targets(idx).boxes, boxes(i,:));
        targets(idx).scores = cat(1, targets(idx).scores, scores(i));
        targets(idx).end = t;
      end
    end    
  end
         
  % ---------------------------------------------------------
  % Detection Guidance Tracking
  % ---------------------------------------------------------
  idx_dets = logical(dets(:,1) == t);
  idx_targets = find([targets.end] == t);
  num_targets = numel(idx_targets);   
  num_dets = sum(idx_dets);
  if num_dets > 0
    det.boxes = round(dets(idx_dets,2:5));    
    det.boxes(:,1) = max(1,det.boxes(:,1));
    det.boxes(:,2) = max(1,det.boxes(:,2));
    det.boxes(:,3) = min(im_w,det.boxes(:,3));
    det.boxes(:,4) = min(im_h,det.boxes(:,4));
    det.boxes(:,3) = det.boxes(:,3)-det.boxes(:,1)+1;
    det.boxes(:,4) = det.boxes(:,4)-det.boxes(:,2)+1; 
    det.scores = dets(idx_dets,6);   
    det.states = zeros(num_dets,1);  
   
    if num_targets > 0      
      % Matching between targets and detections
      [dist, overlap] = ComputeDistTable(conf, im, targets(idx_targets), det.boxes);
      [matching, ~] = Hungarian(dist);    

      % Update targets with detection guidance
      for k = 1:num_targets
        i = idx_targets(k);
        if sum(matching(k,:)) > 0
          j = find(matching(k,:) == 1);
          if det.scores(j) > targets(i).scores(end)
            targets(i).boxes(end,:) = det.boxes(j,:);
            targets(i).scores(end) = det.scores(j);
          end
          det.states(j) = 1;
        end      
      end
      
      % Remove detections
      idx_unmatched = find(det.states == 0);
      num_unmatched = numel(idx_unmatched);
      for k = 1:num_unmatched
        j = idx_unmatched(k);
        max_overlap = max(overlap(:,j));
        if max_overlap > conf.iou_th
          det.states(j) = 1;
        end        
      end  
    end
        
    % Add New Target
    idx_new_targets = find(det.states == 0);  
    num_new_targets = numel(idx_new_targets);
    if num_new_targets > 0
      for i = 1:num_new_targets            
        targets(next_id).id = next_id;
        targets(next_id).start = t;
        targets(next_id).end = t;
        targets(next_id).boxes = det.boxes(idx_new_targets(i),:);
        targets(next_id).scores = det.scores(idx_new_targets(i));
        targets(next_id).model = [];
        targets(next_id).classifier = StructuralSVM(conf.ssvm_opts);
        next_id = next_id + 1;
      end    
    end      
  end
  
  % ---------------------------------------------------------
  % Update Target Model 
  % ---------------------------------------------------------   
  idx_targets = find([targets.end] == t);
  num_targets = numel(idx_targets);
  for i = 1:num_targets
    idx = idx_targets(i);
    targets(idx).model = ...
      UpdateTargetModel(im, targets(idx), targets(idx).end-targets(idx).start+1);
  end  
  
  % ---------------------------------------------------------
  % Display tracking results
  % ---------------------------------------------------------
  DisplayResults(im, targets(idx_targets), t);      
  
end

targets = targets(1:next_id-1);

end


% -----------------------------------------------------------------------------
function targets = BackwardPath(conf, info, net, targets)
% -----------------------------------------------------------------------------
trajectories = CreateTrajectories(info, targets);
for i = 1:numel(targets)
  targets(i).model = targets(i).classifier.get_model();
end

for t = numel(info.im_list):-1:1  
  fprintf('[BackwardPath] frame: %d / %d\n',t,numel(info.im_list));
  im = imread(fullfile(info.im_path, info.im_list{t}));
  [im_h,im_w,~] = size(im);
  % ---------------------------------------------------------
  % Load Forward Path Results
  % ---------------------------------------------------------  
  boxes_forward = trajectories{t}(:,2:5);
   
  % ---------------------------------------------------------
  % Tracking - new targets
  % ---------------------------------------------------------
  idx_targets = find([targets.start] == t+1);
  num_targets = numel(idx_targets);
  if num_targets > 0
    boxes_backward = zeros(num_targets, 4);    
    for i = 1:num_targets    
      idx = idx_targets(i);       
      [boxes_backward(i,:),~] = ...
        ForwardTracking(im, targets(idx).boxes(1,:), targets(idx).model);    
    end 
    
    [~, scores] = GetDetectionScore(conf, im, net, boxes_backward);
    states_detection = logical(scores > conf.lo_th)';
    if size(boxes_forward,1) > 0
      states_existance = CheckExistance(conf, boxes_backward, boxes_forward);
      states = states_detection & states_existance;
    else
      states = states_detection;
    end
    
    for i = 1:num_targets 
      idx = idx_targets(i);
      if states(i) == 1
        targets(idx).boxes = cat(1, boxes_backward(i,:), targets(idx).boxes);
        targets(idx).start = t;
        targets(idx).model = UpdateTargetModel(im, targets(idx), 1);
      end
    end    
  end
  
  % ---------------------------------------------------------
  % Tracking - state update
  % ---------------------------------------------------------      
  states_active = false(1,numel(targets));
  idx_targets = find(([targets.end] > t) & ([targets.start] <= t));
  states_active(idx_targets) = 1;
  for idx = idx_targets
    tt = t-targets(idx).start+1;
    size_prev = targets(idx).boxes(tt+1,3)*targets(idx).boxes(tt+1,4);
    size_curr = targets(idx).boxes(tt,3)*targets(idx).boxes(tt,4);
    if size_prev == size_curr
      states_active(idx) = 0;
    end
    %targets(idx).model = UpdateTargetModel(im, targets(idx), tt+1);
  end
  
  idx_targets = find(states_active == 1);
  num_targets = numel(idx_targets);
  if num_targets > 0
    boxes_backward = zeros(num_targets, 4);    
    for i = 1:num_targets    
      idx = idx_targets(i);  
      tt = t-targets(idx).start+1;
      
      cx = targets(idx).boxes(tt,1)+0.5*targets(idx).boxes(tt,3);
      cy = targets(idx).boxes(tt,2)+0.5*targets(idx).boxes(tt,4);
      xmin = max(1,round(cx-0.5*targets(idx).boxes(tt+1,3)));
      ymin = max(1,round(cy-0.5*targets(idx).boxes(tt+1,4)));
      xmax = min(im_w,round(cx+0.5*targets(idx).boxes(tt+1,3)));
      ymax = min(im_h,round(cy+0.5*targets(idx).boxes(tt+1,4)));
      w = xmax-xmin+1;
      h = ymax-ymin+1;
      center = [xmin,ymin,w,h];

      [boxes_backward(i,:),~] = BackwardTracking(im, center, targets(idx).model);          
    end 
    
    [~, scores] = GetDetectionScore(conf, im, net, boxes_backward);
    for i = 1:num_targets
      idx = idx_targets(i); 
      tt = t-targets(idx).start+1;   
      score_curr = scores(i);
      score_prev = targets(idx).scores(tt);
      if score_curr > score_prev
        targets(idx).boxes(tt,:) = boxes_backward(i,:);  
        targets(idx).model = UpdateTargetModel(im, targets(idx), tt);
      end
    end  
  end  
  
  % ---------------------------------------------------------
  % Display tracking results
  % ---------------------------------------------------------
  idx_targets = find(([targets.end] >= t) & ([targets.start] <= t));
  DisplayResults(im, targets(idx_targets), t);    
end

end

% -----------------------------------------------------------------------------
function trajectories = CreateTrajectories(info, targets)
% -----------------------------------------------------------------------------
num_targets = numel(targets);
num_frames = numel(info.im_list);
trajectories = cell(1,num_frames);
for t = 1:num_frames
  trajectories{t} = zeros(0,5);
end

for i = 1:num_targets
  for t = targets(i).start:targets(i).end
    idx = t-targets(i).start+1;
    trajectories{t} = cat(1, trajectories{t}, [targets(i).id, targets(i).boxes(idx,:)]);
  end
end

end

% -----------------------------------------------------------------------------
function [box, feature] = ForwardTracking(im, center, model)
% -----------------------------------------------------------------------------
search_range = sqrt(center(3)*center(4));
stride = max(2,round(search_range/20));

samples = ExtractForwardTestSamples1(im, center, stride, search_range);
features = ExtractFeatures(im, samples);
scores = StructuredSVMTest(model, features);

[~, idx] = max(scores);
center = samples(idx, :);

search_range = stride;
samples = ExtractForwardTestSamples2(im, center, search_range);
features = ExtractFeatures(im, samples);
scores = StructuredSVMTest(model, features);

[~, idx] = max(scores);
box = samples(idx, :);
feature = features(:, idx);

end

% -----------------------------------------------------------------------------
function [box, feature] = BackwardTracking(im, center, model)
% -----------------------------------------------------------------------------
search_range = sqrt(center(3)*center(4));
stride = max(2,round(search_range/20));

search_range = stride;
samples = ExtractBackwardTestSamples(im, center, search_range);
features = ExtractFeatures(im, samples);
scores = StructuredSVMTest(model, features);

[~, idx] = max(scores);
box = samples(idx, :);
feature = features(:, idx);

end


% % -----------------------------------------------------------------------------
% function [box, feature] = TrackTargetBox(im, center, model)
% % -----------------------------------------------------------------------------
% search_range = sqrt(center(3)*center(4));
% stride = max(2,round(search_range/20));
% 
% samples = ExtractTestSamples(im, center, stride, search_range);
% features = ExtractFeatures(im, samples);
% scores = StructuredSVMTest(model, features);
% 
% [~, idx] = max(scores);
% box = samples(idx, :);
% feature = features(:, idx);
% 
% end


% -----------------------------------------------------------------------------
function [pred_boxes, scores] = GetDetectionScore(conf, im, net, boxes)
% -----------------------------------------------------------------------------
boxes(:,3) = boxes(:,1)+boxes(:,3)-1;
boxes(:,4) = boxes(:,2)+boxes(:,4)-1;

[pred_boxes, scores] = frcn_detect(conf, net, im, boxes, 128);
[im_h,im_w,~] = size(im); 
pred_boxes = round(pred_boxes);
pred_boxes(:,1) = max(1,pred_boxes(:,1));
pred_boxes(:,2) = max(1,pred_boxes(:,2));
pred_boxes(:,3) = min(im_w,pred_boxes(:,3));
pred_boxes(:,4) = min(im_h,pred_boxes(:,4));
pred_boxes(:,3) = pred_boxes(:,3)-pred_boxes(:,1)+1;
pred_boxes(:,4) = pred_boxes(:,4)-pred_boxes(:,2)+1;

end

% -----------------------------------------------------------------------------
function states = CheckOcclusion(conf, boxes, scores)
% -----------------------------------------------------------------------------
num_boxes = size(boxes,1);
states = true(1,num_boxes);
for i = 1:num_boxes
  if states(i) == 0, continue; end  
  overlap = ComputeOverlapRatio(boxes(i,:), boxes);
  overlap(i) = 0;
  idx_overlap = find(overlap > conf.iou_th);
  for k = 1:numel(idx_overlap)
    j = idx_overlap(k);
    if states(j) == 0, continue; end
    if scores(i) > scores(j)
      states(j) = 0;
    else
      states(i) = 0;
    end
  end     
end

end

% -----------------------------------------------------------------------------
function states = CheckExistance(conf, boxes_b, boxes_f)
% -----------------------------------------------------------------------------
num_boxes_b = size(boxes_b,1);
states = true(1,num_boxes_b);
for i = 1:num_boxes_b
  overlap = ComputeOverlapRatio(boxes_b(i,:), boxes_f);
  idx_overlap = find(overlap > conf.iou_th);
  if numel(idx_overlap) > 0
    states(i) = 0;
  end
end


end

% -----------------------------------------------------------------------------
function model = UpdateTargetModel(im, target, idx)
% -----------------------------------------------------------------------------
samples.box = ExtractTrainSamples(im, target.boxes(idx,:));
samples.feature = ExtractFeatures(im, samples.box);
samples.box = samples.box - repmat([samples.box(1,1), samples.box(1,2), 0, 0], [size(samples.box,1),1]);
model = target.classifier.train(samples);

end

% -----------------------------------------------------------------------------
function [dist, overlap] = ComputeDistTable(conf, im, targets, detections)
% -----------------------------------------------------------------------------
num_targets = numel(targets);
num_detections = size(detections,1);
overlap = zeros(num_targets, num_detections);
dist = inf(num_targets, num_detections);
d_features = ExtractFeatures(im, detections);
for i = 1:num_targets   
  t_features = ExtractFeatures(im, targets(i).boxes(end,:));
  overlap(i,:) = ComputeOverlapRatio(targets(i).boxes(end,:), detections); 
  for j = 1:num_detections
    if overlap(i,j) > conf.iou_th
      dist(i,j) = norm(t_features-d_features(:,j));
    end
  end
end

end

% -----------------------------------------------------------------------------
function samples = ExtractForwardTestSamples1(im, center, stride, search_range)
% -----------------------------------------------------------------------------
[im_h, im_w, ~] = size(im);

h_vec = (0:stride:search_range) - round(0.5*search_range);
w_vec = (0:stride:search_range) - round(0.5*search_range);
[w_mat, h_mat] = meshgrid(w_vec, h_vec);

num_temp = numel(h_vec) * numel(w_vec);
temp = repmat(center, [num_temp, 1]);
temp(:, 1) = temp(:, 1) + w_mat(:);
temp(:, 2) = temp(:, 2) + h_mat(:);

one = ones(num_temp, 1);
idx = logical(temp(:,1) > 0 & temp(:,1)+temp(:,3)-one <= im_w & ...
              temp(:,2) > 0 & temp(:,2)+temp(:,4)-one <= im_h );
samples = temp(idx,:);

if isempty(samples)
  stride = 1;
  search_range = 30;
  h_vec = (0:stride:search_range) - round(0.5*search_range);
  w_vec = (0:stride:search_range) - round(0.5*search_range);
  [w_mat, h_mat] = meshgrid(w_vec, h_vec);

  num_temp = numel(h_vec) * numel(w_vec);
  temp = repmat(center, [num_temp, 1]);
  temp(:, 1) = temp(:, 1) + w_mat(:);
  temp(:, 2) = temp(:, 2) + h_mat(:);

  idx = logical(temp(:,1) > 0 & temp(:,1)+temp(:,3)-1 <= im_w & ...
                temp(:,2) > 0 & temp(:,2)+temp(:,4)-1 <= im_h );
  samples = temp(idx,:);  
end

end

% -----------------------------------------------------------------------------
function samples = ExtractForwardTestSamples2(im, center, search_range)
% -----------------------------------------------------------------------------
[im_h, im_w, ~] = size(im);

h_vec = (0:1:search_range) - round(0.5*search_range);
w_vec = (0:1:search_range) - round(0.5*search_range);
[w_mat, h_mat] = meshgrid(w_vec, h_vec);

num_temp = numel(h_vec) * numel(w_vec);
temp = repmat(center, [num_temp, 1]);
temp(:, 1) = temp(:, 1) + w_mat(:);
temp(:, 2) = temp(:, 2) + h_mat(:);

one = ones(num_temp, 1);
idx = logical(temp(:,1) > 0 & temp(:,1)+temp(:,3)-one <= im_w & ...
              temp(:,2) > 0 & temp(:,2)+temp(:,4)-one <= im_h );
samples = temp(idx,:);

end

% -----------------------------------------------------------------------------
function samples = ExtractBackwardTestSamples(im, center, search_range)
% -----------------------------------------------------------------------------
[im_h, im_w, ~] = size(im);

h_vec = (0:1:search_range) - round(0.5*search_range);
w_vec = (0:1:search_range) - round(0.5*search_range);
[w_mat, h_mat] = meshgrid(w_vec, h_vec);

num_temp = numel(h_vec) * numel(w_vec);
temp = repmat(center, [num_temp, 1]);
temp(:, 1) = temp(:, 1) + w_mat(:);
temp(:, 2) = temp(:, 2) + h_mat(:);

one = ones(num_temp, 1);
idx = logical(temp(:,1) > 0 & temp(:,1)+temp(:,3)-one <= im_w & ...
              temp(:,2) > 0 & temp(:,2)+temp(:,4)-one <= im_h );
samples = temp(idx,:);

end


% % -----------------------------------------------------------------------------
% function samples = ExtractTestSamples(im, center, stride, search_range)
% % -----------------------------------------------------------------------------
% [im_h, im_w, ~] = size(im);
% 
% h_vec = (0:stride:search_range) - round(0.5*search_range);
% w_vec = (0:stride:search_range) - round(0.5*search_range);
% [w_mat, h_mat] = meshgrid(w_vec, h_vec);
% 
% num_temp = numel(h_vec) * numel(w_vec);
% temp = repmat(center, [num_temp, 1]);
% temp(:, 1) = temp(:, 1) + w_mat(:);
% temp(:, 2) = temp(:, 2) + h_mat(:);
% 
% one = ones(num_temp, 1);
% idx = logical(temp(:,1) > 0 & temp(:,1)+temp(:,3)-one <= im_w & ...
%               temp(:,2) > 0 & temp(:,2)+temp(:,4)-one <= im_h );
% samples = temp(idx,:);
% 
% if isempty(samples)
%   stride = 1;
%   search_range = 30;
%   h_vec = (0:stride:search_range) - round(0.5*search_range);
%   w_vec = (0:stride:search_range) - round(0.5*search_range);
%   [w_mat, h_mat] = meshgrid(w_vec, h_vec);
% 
%   num_temp = numel(h_vec) * numel(w_vec);
%   temp = repmat(center, [num_temp, 1]);
%   temp(:, 1) = temp(:, 1) + w_mat(:);
%   temp(:, 2) = temp(:, 2) + h_mat(:);
% 
%   idx = logical(temp(:,1) > 0 & temp(:,1)+temp(:,3)-1 <= im_w & ...
%                 temp(:,2) > 0 & temp(:,2)+temp(:,4)-1 <= im_h );
%   samples = temp(idx,:);  
% end
% 
% end

% -----------------------------------------------------------------------------
function samples = ExtractTrainSamples(im, center)
% -----------------------------------------------------------------------------
[im_h, im_w, ~] = size(im);

num_r = 5;
num_t = 16;
rstep = sqrt(center(3)*center(4)) / num_r;
tstep = 2*pi / num_t;

count = 1;
samples = repmat(center, [num_r*num_t+1, 1]);
for i = 1:num_r
  phase = mod(i,2)*0.5*tstep;
  for j = 0:num_t-1
    dx = i*rstep*cos(j*tstep+phase);
    dy = i*rstep*sin(j*tstep+phase);    
    sample = round([center(1)+dx, center(2)+dy, center(3), center(4)]);
    if sample(1) > 0 && sample(1)+sample(3)-1 <= im_w && ...
       sample(2) > 0 && sample(2)+sample(4)-1 <= im_h
      count = count + 1;
      samples(count, :) = sample;
    end    
  end
end
samples = samples(1:count,:);

end
