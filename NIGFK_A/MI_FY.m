function [FY_SU] = MI_FY(dat,anc)
% BATCH_FEATURE_LABEL_FUZZY_SU_ANCHOR_MXM
% 完全依赖锚点计算【特征-标签】模糊互信息
%
%   只计算 m 个锚点之间的相似关系。
%   相似关系矩阵维度: m x m。
%   利用原始数据的 std 来确定高斯核带宽。

X_raw = dat.X_view;
Anchors = anc;
Y_raw = double(dat.label);

% 2. 严格切分前 80% (仅用于计算 std)
n_total = size(Y_raw, 1);
n_train = floor(0.8 * n_total);

Y_train = Y_raw(1:n_train, :);

num_views = length(X_raw);
X_train = cell(1, num_views);
for v = 1:num_views
    feat = double(X_raw{v});
    X_train{v} = feat(1:n_train, :);
end

% 锚点数据 (计算核心)
U_views = Anchors.View;
U_labels = Anchors.Label; % m x L (软标签)

m_anchors = size(U_views{1}, 1);
[~, num_labels] = size(Y_train);

fprintf('   锚点数(m): %d (矩阵维度 m x m)\n', m_anchors);

% ======================================================
% 3. 预计算所有标签的模糊关系矩阵 R_y (m x m)
% ======================================================
% 这里的 R_y 是锚点标签之间的相似度

fprintf('   > 预计算标签的高斯核相似度 R_y (m x m)...\n');
R_y_list = cell(1, num_labels);
H_labels = zeros(1, num_labels);

for l = 1:num_labels
    u_label_vec = U_labels(:, l); % m x 1 (锚点标签值)
    y_vec_raw = Y_train(:, l);    % N x 1 (原始标签值，仅用于算 std)

    % [关键] 使用原始数据计算带宽 delta
    delta_y = std(y_vec_raw);
    if delta_y == 0, delta_y = 1; end

    % 计算 m x m 距离: (u_j - u_k)^2
    diff_mat = abs(u_label_vec - u_label_vec');
    dist_sq = diff_mat.^2;

    % 标签模糊相似度 R_y (m x m)
    R_y = exp(-dist_sq / (2 * delta_y^2));
    R_y_list{l} = R_y;

    % 计算标签熵 H(Y) - 基于 m 个锚点
    card_y = sum(R_y, 2);
    H_labels(l) = -mean(log2(card_y / m_anchors));
end

% 结果容器
FY_SU = cell(1, num_views);

% ======================================================
% 4. 遍历视图计算 SU (特征 vs 标签)
% ======================================================
for v = 1:num_views
    fprintf('  > 视图 %d 计算中...\n', v);

    X_v = X_train{v};      % N x d (用于算 std)
    U_v = U_views{v};      % m x d (用于算相似度)
    [~, n_features] = size(X_v);

    SU_matrix = zeros(n_features, num_labels);

    for i = 1:n_features
        u_vec = U_v(:, i);   % m x 1 (锚点特征)
        f_vec_raw = X_v(:, i); % N x 1 (原始特征)

        % --- A. 计算特征模糊关系 R_f (m x m) ---
        % [关键] 使用原始数据计算带宽
        delta_f = std(f_vec_raw);
        if delta_f == 0, delta_f = 1; end

        % 计算 m x m 距离
        diff_mat = abs(u_vec - u_vec');
        dist_sq = diff_mat.^2;

        % 特征模糊相似度 R_f (m x m)
        R_f = exp(-dist_sq / (2 * delta_f^2));

        % 特征熵 H(F)
        card_f = sum(R_f, 2);
        H_f = -mean(log2(card_f / m_anchors));

        % --- B. 遍历标签计算互信息 ---
        row_su = zeros(1, num_labels);

        for l = 1:num_labels
            % 取出预计算的 R_y (m x m) 和 H(Y)
            R_y = R_y_list{l};
            H_Y = H_labels(l);

            % --- C. 联合模糊关系 (Min Operator) ---
            % 两个 m x m 矩阵的交集
            R_joint = min(R_f, R_y);

            % --- D. 联合熵 H(F, Y) ---
            card_joint = sum(R_joint, 2);
            H_joint = -mean(log2(card_joint / m_anchors));

            % --- E. 计算 SU ---
            MI = H_f + H_Y - H_joint;
            denom = H_f + H_Y;

            val = 0;
            if denom > 1e-10
                val = 2 * MI / denom;
            end
            row_su(l) = max(0, min(1, val));
        end
        SU_matrix(i, :) = row_su;
    end
    FY_SU{v} = SU_matrix;
end

fprintf('全部完成。\n');
end