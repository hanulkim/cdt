classdef StructuralSVM < handle
  %% Properties
  properties
    MAX_ITER;
    MAX_N_SV;   
    MAX_N_SP;
    FEATURE_DIM;
    C;
    K;
    model;
    svs;  
    sps;
    sps_check;
  end
  %% Methods
  methods
    %% Abstract Level 0 Methods
    function obj = StructuralSVM(opts)
      obj.MAX_ITER = opts.MAX_ITER;
      obj.MAX_N_SV = opts.MAX_N_SV;
      obj.MAX_N_SP = opts.MAX_N_SP;
      obj.FEATURE_DIM = opts.FEATURE_DIM;
      obj.C = opts.C;
      obj.K = zeros(obj.MAX_N_SV+2, obj.MAX_N_SV+2);       
      
      obj.model = zeros(obj.FEATURE_DIM, 1);      
      obj.svs.feature = zeros(obj.FEATURE_DIM, obj.MAX_N_SV+1);
      obj.svs.overlap = zeros(1, obj.MAX_N_SV+1); 
      obj.svs.weight = zeros(1, obj.MAX_N_SV+1);
      obj.svs.pattern_id = zeros(1, obj.MAX_N_SV+1);
      obj.svs.data_id = zeros(1, obj.MAX_N_SV+1);
      obj.svs.gradient = zeros(1, obj.MAX_N_SV+1);
      obj.svs.num = 0;
      
      obj.sps = cell(1, obj.MAX_N_SP+1);
      obj.sps_check = zeros(1, obj.MAX_N_SP+1);
    end
    
    function score = test(obj, dataset)
      score = obj.model' * dataset;
    end    
    
    function model = train(obj, samples)          
      pattern_id = find(obj.sps_check == 0, 1);  
      obj.sps{pattern_id} = samples;      
      obj.sps{pattern_id}.ref = 0;
      obj.sps_check(pattern_id) = 1;
      
      obj.process_new(pattern_id);
      obj.update_svs();      
      
      for i = 1:obj.MAX_ITER
        obj.re_process();
        obj.update_svs();        
      end
      
      obj.compute_model();
      
      % test
%       t_overlap = obj.loss(samples.box);
%       t_score = obj.score(samples.feature);
%       t_loss = 0;
%       for i = 2:size(samples.box,1)
%         d_score = t_score(1) - t_score(i);
%         t_loss = t_loss + max(0, t_overlap(i) - d_score);
%       end
%       t_loss = t_loss / size(samples.box,1);
%       t_cost = 0.5*0.001*obj.model' * obj.model + t_loss; 
%       fprintf('[After] cost: %.8f, loss: %.8f\n', t_cost, t_loss);      
      
      model = obj.model;
      
    end
    
    function model = get_model(obj)
      model = obj.model;
    end
    
    %% 
    function process_new(obj, pattern_id)
      if obj.svs.num == 0
        obj.model = zeros(obj.FEATURE_DIM, 1); 
      else
        obj.compute_model();      
      end
      
      idx = 1;
      val = -obj.score(obj.sps{pattern_id}.feature(:,idx));
      ip = obj.add_sv(pattern_id, idx, val);
      
      [idx, val] = obj.min_gradient(pattern_id);
      in = obj.add_sv(pattern_id, idx, val);
      
      obj.smo_step(ip, in);
    end
    
    function update_svs(obj)
      while obj.svs.num > obj.MAX_N_SV
        min_val = inf;
        for i = 1:obj.svs.num
          if obj.svs.weight(i) < 0
            j = find(obj.svs.weight(1:obj.svs.num) > 0 & ...
                     obj.svs.pattern_id(1:obj.svs.num) == obj.svs.pattern_id(i), ...
                     1);
            val = (obj.svs.weight(i)^2) * (obj.K(i,i) + obj.K(j,j) - 2*obj.K(i,j));
            if val < min_val
              min_val = val;
              in = i;
              ip = j;
            end         
          end
        end
        obj.svs.weight(ip) = obj.svs.weight(ip) + obj.svs.weight(in);
        obj.remove_sv(in);
        if ip > obj.svs.num
          ip = in;
        end        
        if obj.svs.weight(ip) < 1e-8
          obj.remove_sv(ip);
        end 
        
        obj.compute_model();
        loss = obj.svs.overlap(1:obj.svs.num);
        score = obj.score(obj.svs.feature(:, 1:obj.svs.num)); 
        obj.svs.gradient(1:obj.svs.num) = -(loss+score);         
      end      
    end    
    
    function re_process(obj)
      obj.process_old();
      for i = 1:obj.MAX_ITER
        obj.optimize();
      end
    end    

    function process_old(obj)
      if obj.svs.num == 0
        return;
      end
      
      pattern_id = find(obj.sps_check > 0);
      pattern_id = pattern_id(randi(size(pattern_id,1)));
      
      ip = -1; 
      max_grad = -inf;
      sv_id = find(obj.svs.pattern_id(1:obj.svs.num) == pattern_id);
      for k = 1:length(sv_id)
        i = sv_id(k);
        if obj.svs.gradient(i) > max_grad && obj.svs.weight(i) < obj.C * (obj.svs.data_id(i) == 1)
          ip = i;
          max_grad = obj.svs.gradient(i);
        end
      end   
      if ip == -1
        return;
      end
      
      [idx, val] = obj.min_gradient(pattern_id);
      in = find(obj.svs.pattern_id(1:obj.svs.num) == pattern_id & ...
                obj.svs.data_id(1:obj.svs.num) == idx, 1);
      if isempty(in)
        in = obj.add_sv(pattern_id, idx, val);
      end
      
      obj.smo_step(ip, in);
    end
    
    function optimize(obj)
      if sum(obj.sps_check) == 0
        return;
      end
      
      pattern_id = find(obj.sps_check > 0);
      pattern_id = pattern_id(randi(size(pattern_id,1)));
      
      max_grad = -inf;
      min_grad = inf;
      sv_id = find(obj.svs.pattern_id(1:obj.svs.num) == pattern_id);
      for k = 1:length(sv_id)
        id = sv_id(k); 
        if obj.svs.gradient(id) > max_grad && obj.svs.weight(id) < obj.C * (obj.svs.pattern_id(id) == 1)
          ip = id;
          max_grad = obj.svs.gradient(id);
        end
        if obj.svs.gradient(id) < min_grad
          in = id;
          min_grad = obj.svs.gradient(id);
        end        
      end
      
      obj.smo_step(ip, in);
    end      
    
    %% 
    function smo_step(obj, ip, in)
      if ip == in
        return; 
      end
      
      d_gradient = obj.svs.gradient(ip) - obj.svs.gradient(in); 
      if d_gradient >= 1e-5
        ks = obj.K(ip, ip) + obj.K(in, in) - 2*obj.K(ip, in);
        loss_ubound = d_gradient / ks;
        indicator = (obj.svs.data_id(ip) == 1);
        loss = min(loss_ubound, obj.C * indicator - obj.svs.weight(ip));
        
        obj.svs.weight(ip) = obj.svs.weight(ip) + loss;
        obj.svs.weight(in) = obj.svs.weight(in) - loss;        
        obj.svs.gradient(1:obj.svs.num) = obj.svs.gradient(1:obj.svs.num) - ...
                                          loss*(obj.K(1:obj.svs.num, ip)-obj.K(1:obj.svs.num, in))'; 
      end
      
      if abs(obj.svs.weight(ip)) < 1e-8
        obj.remove_sv(ip);
        if in > obj.svs.num
          in = ip;
        end       
      end
      
      if abs(obj.svs.weight(in)) < 1e-8
        obj.remove_sv(in);
      end   
    end 

    function i = add_sv(obj, pattern_id, idx, val)
      i = obj.svs.num + 1;      
      obj.svs.weight(i) = 0;      
      obj.svs.pattern_id(i) = pattern_id;
      obj.svs.data_id(i) = idx;
      obj.svs.feature(:,i) = obj.sps{obj.svs.pattern_id(i)}.feature(:,obj.svs.data_id(i));       
      obj.svs.overlap(i) = obj.loss(obj.sps{obj.svs.pattern_id(i)}.box(1,:),...
                                    obj.sps{obj.svs.pattern_id(i)}.box(obj.svs.data_id(i),:)); 
      obj.svs.gradient(i) = val;                              
      obj.svs.num = obj.svs.num + 1;
      
      obj.sps{pattern_id}.ref = obj.sps{pattern_id}.ref + 1; 
      
      obj.K(i, 1:i-1) = obj.svs.feature(:, i)' * obj.svs.feature(:, 1:i-1);
      obj.K(1:i-1, i) = obj.K(i, 1:i-1);
      obj.K(i, i) = obj.svs.feature(:, i)' * obj.svs.feature(:, i);
    end

    function remove_sv(obj, idx)
      pattern_id = obj.svs.pattern_id(idx);
      obj.sps{pattern_id}.ref = obj.sps{pattern_id}.ref - 1;
      
      if obj.sps{pattern_id}.ref == 0
        obj.sps{pattern_id} = [];
        obj.sps_check(pattern_id) = 0;
      end     
      
      if idx ~= obj.svs.num
        obj.svs.weight(idx) = obj.svs.weight(obj.svs.num);
        obj.svs.pattern_id(idx) = obj.svs.pattern_id(obj.svs.num);
        obj.svs.data_id(idx) = obj.svs.data_id(obj.svs.num);
        obj.svs.feature(:,idx) = obj.svs.feature(:,obj.svs.num);
        obj.svs.overlap(idx) = obj.svs.overlap(obj.svs.num);
        obj.svs.gradient(idx) = obj.svs.gradient(obj.svs.num);        
        obj.K(idx,:) = obj.K(obj.svs.num,:);
        obj.K(:,idx) = obj.K(:,obj.svs.num);      
      end
      obj.svs.num = obj.svs.num - 1;   
    end    
    
    function [min_grad_idx, min_grad_val] = min_gradient(obj, pattern_id)
      obj.compute_model();
      loss = obj.loss(obj.sps{pattern_id}.box);
      score = obj.score(obj.sps{pattern_id}.feature);
      gradient = -(loss+score);
      [min_grad_val, min_grad_idx] = min(gradient);            
    end    
    
    %% 
    function compute_model(obj)
      obj.model = obj.svs.feature(:,1:obj.svs.num) * obj.svs.weight(1:obj.svs.num)';
    end
    
    function val = score(obj, data)
      val = obj.model' * data;
    end        
    
    function val = loss(~, box, box2)
      if nargin == 3
        x1 = max(box(1), box2(1));
        x2 = min(box(1)+box(3), box2(1)+box2(3));
        y1 = max(box(2), box2(2));
        y2 = min(box(2)+box(4), box2(1)+box(3));

        area1 = box(3) * box(4);
        area2 = box2(3) * box2(4);      
        i_area = (x2-x1)*(y2-y1);
        u_area = area1 + area2 - i_area;      
        overlap = i_area / u_area;

        val = 1.0 - overlap;
      else
        xmin = max(box(1,1), box(:,1));
        xmax = min(box(1,1)+box(1,3), box(:,1)+box(:,3));
        ymin = max(box(1,2), box(:,2));
        ymax = min(box(1,2)+box(1,4), box(:,2)+box(:,4));

        area1 = box(1,3) * box(1,4);
        area2 = box(:,3) .* box(:,4);
        i_area = (xmax-xmin) .* (ymax-ymin);
        u_area = area1 + area2 - i_area;
        overlap = i_area ./ u_area;

        val = (1.0 - overlap)';        
      end   
    end
  end
end