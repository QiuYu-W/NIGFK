
function [Anchors, ModelInfo]=Anchor_MVML(data)
    % =================================================================
    % 批量多视图多标签锚点生成 (定制版)
    % 修改点：
    %   1. 标签变量名为 'label'
    %   2. 移除特征归一化 (直接使用)
    %   3. 自定义锚点数量策略 (5*C 或 3*C)
    % =================================================================
         
        Y_raw = double(data.label);
 
        X_raw = data.X_view;

        % 2. 数据切分 (只取前 80%)
        num_total = size(Y_raw, 1);
        num_train = floor(0.8 * num_total);
        
        if num_train < 10
            warning('样本数过少 (%d)，跳过', num_train);
        end
        
        % 切分标签
        Y_train = Y_raw(1:num_train, :);
        
        % 切分视图数据
        num_views = length(X_raw);
        X_train = cell(1, num_views);
        for v = 1:num_views
            feat = double(X_raw{v});
            X_train{v} = feat(1:num_train, :);
        end
        
        % 3. 确定锚点数量 m (你的自定义策略)
        num_classes = size(Y_train, 2);
        
        if num_classes < 200
            m_target = 5 * num_classes;
        else
            m_target = 3 * num_classes;
        end
        
        % [安全锁]：锚点数量不能超过样本数量！
        % 如果计算出的 m 比样本还多，litekmeans 会报错。
        % 这里限制为样本数的 90% 或者 m_target 的较小值
        m = min(m_target, floor(0.9 * num_train));
        
        % 再次确保 m 至少为 1 (防止极端情况)
        m = max(m, 1);
        
        % 4. 核心：生成锚点
        t_start = tic;
        [Anchors, ModelInfo] = train_joint_anchors_lite_custom(X_train, Y_train, m);
        t_cost = toc(t_start);
        
        fprintf('   锚点生成完成，耗时: %.2f 秒 (Lambda=%.4f)\n', t_cost, ModelInfo.lambda);
        
        % 5. 保存结果
        % save_name = [name_no_ext, '_Anchors.mat'];
        % save(fullfile(out_path, save_name), 'Anchors', 'ModelInfo', 'm');
        
    fprintf('------------------------------------------------\n');
    fprintf('所有数据集处理完毕。\n');
end


%% =============================================================
%  子函数：基于 LiteKMeans 的联合锚点学习 (去归一化版)
% =============================================================
function [Anchors, ModelInfo] = train_joint_anchors_lite_custom(X_cells, Y, m)
    num_views = length(X_cells);
    view_dims = zeros(1, num_views);
    
    % --- Step 1: 视图特征拼接 (不进行 Min-Max) ---
    for v = 1:num_views
        view_dims(v) = size(X_cells{v}, 2);
    end
    
    % 直接拼接
    X_concat = cat(2, X_cells{:});
    
    % --- Step 2: 自动计算 Lambda ---
    % 依然需要计算能量平衡，防止特征数值和0/1标签数值量级差异太大
    norm_X = norm(X_concat, 'fro');
    norm_Y = norm(Y, 'fro');
    
    if norm_Y == 0
        lambda = 1; 
    else
        lambda = norm_X / norm_Y;
    end
    
    % --- Step 3: 构建联合空间 Z ---
    Z = [X_concat, lambda * Y];
    
    % --- Step 4: 运行 LiteKMeans ---
    % 'Replicates', 20 保证小样本下的稳定性
    [~, Centers_Z] = litekmeans(Z, m, 'Replicates', 20, 'MaxIter', 100);
    
    % --- Step 5: 分离结果 ---
    Anchors.View = cell(1, num_views);
    current_idx = 0;
    
    % 提取各视图锚点
    for v = 1:num_views
        dim = view_dims(v);
        idx_range = (current_idx + 1) : (current_idx + dim);
        Anchors.View{v} = Centers_Z(:, idx_range);
        current_idx = current_idx + dim;
    end
    
    % 提取并还原标签锚点
    Anchors.Label_Scaled = Centers_Z(:, current_idx+1 : end);
    Anchors.Label = Anchors.Label_Scaled / lambda;
    
    % 保存模型参数
    ModelInfo.lambda = lambda;
    ModelInfo.view_dims = view_dims;
end