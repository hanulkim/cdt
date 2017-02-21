function trajoriess = GenerateTrajectories(config, seq_idx, opts)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2015, Hanul Kim
% 
% This file is part of the MOT code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Load Infomation
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lib_path = 'D:\OpenSource\matconvnet-1.0-beta17\matconvnet-1.0-beta17\matlab';
run(fullfile(lib_path,'vl_setupnn.m'));
            
color_table = LoadColorTable();
seq_info = LoadSeqInfo(config, seq_idx);
convnet_model = load(config.convnet_model_fn);
activation_model = importdata(config.activation_model_fn);
regression_model = importdata(config.regression_model_fn);

convnet_offset = convnet_model.meta.normalization.averageImage;
convnet_offsets = cat(3,...
                 repmat(convnet_offset(1), [224,224]),...
                 repmat(convnet_offset(2), [224,224]),...
                 repmat(convnet_offset(3), [224,224]));
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Generate Intial Trajectories
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
max_active_targets = opts.max_active_target;
targets_info.next_id = 1;
targets_info.state = zeros(max_active_targets, 1);
targets = struct('id',[],'state',[],'pos',[],'model',[]);
targets(max_active_targets).id = [];

trajoriess = cell(1,seq_info.num_frame);
for frame_id = 1:seq_info.num_frame  
  img = imread(fullfile(seq_info.img_path, seq_info.img_list{frame_id}));
  [img_h, img_w,~] = size(img);
  
  % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  % Tracking
  % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  active_idx = find(targets_info.state == 1);
  num_target = numel(active_idx);
  for i = 1:num_target    
    idx = active_idx(i);  
    [targets(idx).pos, targets(idx).score, targets(idx).time] = TrackTarget(img, targets(idx));
  end  
  
  % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
  % Regularization with Detection Guidance
  % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  % Collect active detections
  num_detections = seq_info.det_list{frame_id}.num;
  if num_detections > 0
    detections.pos = seq_info.det_list{frame_id}.pos;
    detections.state = zeros(num_detections, 1);  
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    features = zeros(4096,num_detections);
    data = single(zeros(224,224,3,num_detections));
    for i = 1:num_detections
      x = detections.pos(i,1);
      y = detections.pos(i,2);
      w = detections.pos(i,3);
      h = detections.pos(i,4);
      img_ = img(y:y+h-1,x:x+w-1,:);
      img_ = single(imresize(img_, convnet_model.meta.normalization.imageSize(1:2)));   
      bsxfun(@minus, img_, convnet_offsets);      
      data(:,:,:,i) = img_; 
    end  
    res = vl_simplenn(convnet_model, data);  
    features(:,:) = res(15).x(:,:,:,:); 
    scores = activation_model(1:end-1)'*features + activation_model(end);
    idx = logical(scores > opts.activation_th);
    detections.pos = detections.pos(idx, :);    
    detections.state = detections.state(idx);  
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    detections.feature = ExtractFeatures(img, detections.pos);    
    detections = BoundingBoxRegression(img, detections, regression_model);
    num_detections = numel(detections.state);      
  end
      
  if num_detections > 0 
    % Match trajectories and detections
    overlap_table = zeros(num_target, num_detections);
    dist_table = zeros(num_target, num_detections);
    iou_dist_table = zeros(num_target, num_detections);
    app_dist_table = zeros(num_target, num_detections);
    for i = 1:num_target   
      idx = active_idx(i);
      overlap_table(i,:) = ComputeOverlapRatio(targets(idx).pos, detections.pos); 
      iou_dist_table(i,:) = 1 - overlap_table(i,:);
      unmatched_idx = logical(overlap_table(i,:) < opts.overlap_th);
      iou_dist_table(i,unmatched_idx) = inf;

      model_score = StructuredSVMTest(targets(idx).model,detections.feature);
      app_dist_table(i,:) = 1 ./ (1+exp(model_score));
      dist_table(i,:) = iou_dist_table(i,:) .* app_dist_table(i,:);
    end
    [matched_table,~] = Hungarian(dist_table);
    
    % Scale adapdation trajectories and detections
    for i = 1:num_target
      idx = active_idx(i);
      if sum(matched_table(i,:)) > 0
        j = logical(matched_table(i,:) == 1);      
        iou = overlap_table(i,j);
        alpha = 1/min(10, max(5, targets(idx).time));
        targets(idx).mean_iou = (1-alpha)*targets(idx).mean_iou + alpha*iou;
        detections.state(j) = 1;

        % scale change
        tcx = targets(idx).pos(1) + 0.5*targets(idx).pos(3);
        tcy = targets(idx).pos(2) + 0.5*targets(idx).pos(4);
        tw = targets(idx).pos(3);
        th = targets(idx).pos(4);

        dcx = detections.pos(j,1) + 0.5*detections.pos(j,3);
        dcy = detections.pos(j,2) + 0.5*detections.pos(j,4);
        dw = detections.pos(j,3);
        dh = detections.pos(j,4);

        aa = opts.detection_ratio;

        tcx = (1-aa)*tcx + aa*dcx;
        tcy = (1-aa)*tcy + aa*dcy;

        tw = round((1-aa)*tw + aa*dw);
        th = round((1-aa)*th + aa*dh);
        tx = max(1,round(tcx - 0.5*tw));
        ty = max(1,round(tcy - 0.5*th));
        if tx+tw > img_w
          tw = img_w - tx;
        end
        if ty+th > img_h
          th = img_h - ty;
        end

        targets(idx).pos = [tx,ty,tw,th]; 
      else
        alpha = 1/min(10, max(5, targets(idx).time));
        targets(idx).mean_iou = (1-alpha)*targets(idx).mean_iou;
      end
    end  
    
    % Remove detections
    unmatched_idx = find(detections.state == 0);
    num_unmatched_det = numel(unmatched_idx);
    for i = 1:num_unmatched_det
      idx = unmatched_idx(i);
      if max(overlap_table(:,idx)) > 0.3
        detections.state(idx) = 1;
      end
    end    
    
    % Generation
    unmatched_idx = find(detections.state == 0);  
    num_unmatched_det = numel(unmatched_idx);
    if num_unmatched_det > 0
      inactive_idx = find(targets_info.state == 0, num_unmatched_det);
      for i = 1:num_unmatched_det            
        idx = inactive_idx(i);
        targets(idx).id = targets_info.next_id;
        targets(idx).state = 1;
        targets(idx).pos = detections.pos(unmatched_idx(i),:);
        targets(idx).model = [];
        targets(idx).stride = 2;
        targets(idx).search_range = opts.search_range;
        targets(idx).color = color_table.rgb(color_table.next,:);    
        targets(idx).time = 1;
        targets(idx).mean_iou = 1;
        targets(idx).score = 1;

        targets_info.state(idx) = 1;      
        targets_info.next_id = targets_info.next_id + 1;     

        color_table.next = mod(color_table.next+1, 20);
        if color_table.next == 0
          color_table.next = 1;
        end
      end    
    end    
  end
  
  % Termination
  features = zeros(4096,1);
  for i = 1:num_target
    idx = active_idx(i);    
    x = targets(idx).pos(1);
    y = targets(idx).pos(2);
    w = targets(idx).pos(3);
    h = targets(idx).pos(4);
    img_ = img(y:y+h-1,x:x+w-1,:);
    img_ = single(imresize(img_, convnet_model.meta.normalization.imageSize(1:2)));     
    bsxfun(@minus, img_, convnet_offsets);
    res = vl_simplenn(convnet_model,img_);  
    features(:) = res(15).x(:,:,:);
    scores = activation_model(1:end-1)'*features + activation_model(end);
    if scores < opts.activation_th
      targets(idx).state = 0;
      targets_info.state(idx) = 0;
    end
  end
    
  % Update target models
  active_idx = find(targets_info.state == 1);
  num_active_target = numel(active_idx);
  for i = 1:num_active_target
    idx = active_idx(i);
    targets(idx).model = UpdateTargetModel(img, targets(idx), opts);
  end  
  
  % Save tracking results
  if num_active_target > 0
    trajoriess{frame_id} = zeros(num_active_target,5);
    for i = 1:num_active_target
      idx = active_idx(i);
      trajoriess{frame_id}(i,1) = targets(idx).id;
      trajoriess{frame_id}(i,2:5) = targets(idx).pos;
    end 
  end
  
  % Display tracking results
  if opts.display == true
    DisplayResults(img, targets, active_idx, frame_id);
  end      
end
  
end