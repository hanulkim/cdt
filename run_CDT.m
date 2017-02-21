function [] = run_MOT_V02()
% AUTORIGHTS
% -----------------------------------------------------------------------------
% Copyright (c) 2016, Hanul Kim
% 
% This file is part of the MOT code and is available under the terms of
% the Simplified BSD License provided in LICENSE. Please retain this notice 
% and LICENSE if you use this file (or any portion of it) in your project.
% -----------------------------------------------------------------------------

% -----------------------------------------------------------------------------
% Configuration
% -----------------------------------------------------------------------------
addpath('src');
addpath('data');
addpath('tools');
addpath(fullfile('lib','caffe','matlab'));
addpath(fullfile('lib','fast_rcnn'));

conf = LoadConfig();

% -----------------------------------------------------------------------------
% Caffe Setup
% -----------------------------------------------------------------------------
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_file = fullfile('log', ['test_', timestamp, '.txt']);
diary(log_file);

%gpuDevice([]);
%gpu_id = 1;
%disp(gpuDevice(gpu_id));
%caffe.set_device(gpu_id-1);

net = caffe.Net(conf.net_def, 'test');
net.copy_from(conf.net_model);
%caffe.set_mode_gpu();
caffe.set_mode_cpu();
  
fprintf('###########################################################\n');
fprintf('# Seuence Name: %s\n', conf.seq_info.name);
  
% ---------------------------------------------------------
% Detection
% ---------------------------------------------------------
fprintf('# --- 01. Detection \n');
dets_ = importdata(fullfile(conf.seq_info.det_path));    
det_idx = logical(dets_(:,6) > conf.hi_th);
dets = dets_(det_idx,:);
  
% ---------------------------------------------------------
% Tracking
% ---------------------------------------------------------  
fprintf('# --- 02. Tracking \n');
trajs = Tracking(conf, conf.seq_info, net, dets);
     
% ---------------------------------------------------------
% Refinement
% ---------------------------------------------------------    
fprintf('# --- 04. Refinement \n');
num_frames = numel(trajs);
max_target_id = 0;
for t = 1:num_frames
  if numel(trajs{t}) > 0
    target_id = max(trajs{t}(:,1));
    if target_id > max_target_id
      max_target_id = target_id;
    end      
  end  
end
  
target_length = zeros(max_target_id, 1);
for t = 1:num_frames
  if numel(trajs{t}) > 0
    num_target = size(trajs{t},1);
    for i = 1:num_target
      target_id = trajs{t}(i,1);
      target_length(target_id) = target_length(target_id)+1; 
    end
  end
end     
  
% ---------------------------------------------------------
% Save  
% ---------------------------------------------------------  
fprintf('# --- 05. Save \n');
SaveResults(conf, trajs, n, target_length);
  
fprintf('###########################################################\n\n');

caffe.reset_all(); 
gpuDevice([]);

end


% -----------------------------------------------------------------------------
function conf = LoadConfig()
% -----------------------------------------------------------------------------  
conf.seq_info.name = 'ADL-Rundle-8';
conf.seq_info.im_path = fullfile('data','ADL-Rundle-8_im');
conf.seq_info.im_list = dir(fullfile(conf.seq_info.im_path, '*.jpg'));
conf.seq_info.im_list = {conf.seq_info.im_list.name};
conf.seq_info.det_path = fullfile('data','ADL-Rundle-8_det.mat');

conf.hi_th = 0.99;
conf.lo_th = 0.50; 
conf.iou_th = 0.3;

conf.res_path = fullfile('results');

conf.net_def = fullfile('model','vgg16.prototxt');
conf.net_model = fullfile('model','vgg16_train.caffemodel');

conf.scales = 600;
conf.max_size = 1000;
conf.image_means = importdata('model/mean_image.mat');

conf.ssvm_opts.MAX_ITER = 11;        
conf.ssvm_opts.MAX_N_SV = 100;
conf.ssvm_opts.MAX_N_SP = 100;
conf.ssvm_opts.FEATURE_DIM = 3520;
conf.ssvm_opts.C = 10;   

end

% -----------------------------------------------------------------------------
function seq_info = LoadSeqInfo(conf)
% -----------------------------------------------------------------------------
seq_info = cell(1,numel(conf.seq_list));
for n = 1:numel(conf.seq_list)
  seq_info{n}.name = conf.seq_list{n};
  seq_info{n}.im_path = fullfile(conf.seq_path,'im',seq_info{n}.name);
  seq_info{n}.im_list = dir(fullfile(seq_info{n}.im_path, '*.jpg'));
  seq_info{n}.im_list = {seq_info{n}.im_list.name};
end

end

% -----------------------------------------------------------------------------
function [] = SaveResults(conf, trajs, n, target_length)
% -----------------------------------------------------------------------------
if ~exist(conf.res_path,'dir') 
  mkdir(conf.res_path); 
end

res_fn = sprintf('%s.txt', conf.seq_list{n});
res_fid = fopen(fullfile(conf.res_path,res_fn),'w');
for t = 1:numel(trajs)
  if numel(trajs{t}) == 0, continue; end

  num_target = size(trajs{t},1);
  for i = 1:num_target
    target_id = trajs{t}(i,1);
    if target_length(target_id) < 1, continue; end
    
    fprintf(res_fid,...
           '%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1\n',...
           t,...
           trajs{t}(i,1),...
           trajs{t}(i,2),...
           trajs{t}(i,3),...
           trajs{t}(i,4),...
           trajs{t}(i,5));     
  end
end    
fclose(res_fid);     

end