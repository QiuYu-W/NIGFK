
function [X_K] = MI_FF(dat,anc)
% BATCH_FUZZY_SU_ANCHOR_MXM
% 完全依赖锚点计算【特征-特征】模糊互信息 
%   将数据集视为只有 m 个锚点。
%   特征 $f_i$ 和 $f_j$ 的互信息完全基于它们在 m 个锚点上的分布计算。
%   相似关系矩阵维度: m x m。

X_raw = dat.X_view;
Anchors = anc;

% 获取标签 (仅用于确定 std 计算的样本范围)
Y_raw = double(dat.label);

% 2. 严格切分前 80% (用于计算带宽 Sigma)
n_total = size(Y_raw, 1);
n_train = floor(0.8 * n_total);

num_views = length(X_raw);
X_train = cell(1, num_views);

% 锚点数据 (这是计算核心)
U_views = Anchors.View;

for v = 1:num_views
    feat = double(X_raw{v});
    X_train{v} = feat(1:n_train, :);
end

m_anchors = size(U_views{1}, 1);
fprintf('   锚点数(m): %d (将基于此维度计算相关性)\n', m_anchors);


% 结果容器
X_K = cell(size(X_raw));

% ======================================================
% 3. 遍历每个视图计算特征间 SU
% ======================================================
for v = 1:num_views
    fprintf('  > 视图 %d 计算中...\n', v);

    X_v = X_train{v};      % N x d (仅用于计算 std)
    U_v = U_views{v};      % m x d (计算核心)
    [~, n_features] = size(X_v);

    % --- A. 预计算所有特征的 m x m 模糊矩阵 ---

    fprintf('    预计算特征锚点互相关矩阵 R_all (m x m x d)...\n');

    R_all = zeros(m_anchors, m_anchors, n_features);
    H_list = zeros(n_features, 1);

    for i = 1:n_features
        u_vec = U_v(:, i);   % m x 1 (锚点特征)

        % 带宽 delta 依然建议用原始数据计算，以反映真实分布
        f_vec_raw = X_v(:, i);
        delta = std(f_vec_raw);
        if delta == 0, delta = 1; end

        % 计算 m 个锚点两两之间的距离
        % (m x 1) - (1 x m) -> m x m
        diff_mat = abs(u_vec - u_vec');
        dist_sq = diff_mat.^2;

        % 生成 m x m 的模糊关系矩阵
        R_f = exp(-dist_sq / (2 * delta^2));

        R_all(:, :, i) = R_f;

        % 熵 H(F) - 基于 m 个点
        card = sum(R_f, 2);
        H_list(i) = -mean(log2(card / m_anchors));
    end

    % --- B. 计算两两特征 SU ---
    SU_matrix = zeros(n_features, n_features);

    fprintf('    计算 SU 矩阵 (基于 m x m)...\n');

    for i = 1:n_features
        % 提取特征 i 的 R 矩阵 (m x m)
        R_i = R_all(:, :, i);
        H_i = H_list(i);

        row_su = zeros(1, n_features);

        for j = i+1:n_features
            R_j = R_all(:, :, j);
            H_j = H_list(j);

            % 联合模糊关系 (Min Operator) - 运算量仅为 m^2
            R_joint = min(R_i, R_j);

            % 联合熵
            card_joint = sum(R_joint, 2);
            H_joint = -mean(log2(card_joint / m_anchors));

            % SU 计算
            MI = H_i + H_j - H_joint;
            denom = H_i + H_j;

            val = 0;
            if denom > 1e-10
                val = 2 * MI / denom;
            end
            row_su(j) = max(0, min(1, val));
        end
        SU_matrix(i, :) = row_su;
    end

    % 对称化 + 对角线置1
    SU_matrix = SU_matrix + SU_matrix';
    SU_matrix(logical(eye(n_features))) = 1;


    X_K{v} = SU_matrix;

    % 清理内存
    clear R_all SU_matrix
end

end
