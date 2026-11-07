
function [LL_SU] = MI_YY(dat,anc)
% BATCH_LABEL_LABEL_SU_ANCHOR_MXM
% 利用锚点加速计算【标签-标签】模糊互信息
%
%   只计算 m 个锚点之间的标签相似关系。
%   相似关系矩阵维度: m x m。
%   利用原始数据的 std 来确定高斯核带宽，处理软标签 (Soft Labels)。

Anchors = anc;
Y_raw = double(dat.label);


% 2. 截取前 80% (仅用于计算 std，确定高斯核宽度)
n_total = size(Y_raw, 1);
n_train = floor(0.8 * n_total);

Y_train = Y_raw(1:n_train, :); % N x L
U_labels = Anchors.Label;      % m x L (软标签)

m_anchors = size(U_labels, 1);
num_labels = size(Y_train, 2);

fprintf('   锚点数(m): %d (矩阵维度 m x m)\n', m_anchors);



% ======================================================
% 3. 预计算所有标签的 m x m 模糊矩阵 R_y
% ======================================================
% 这里的 R_y 是锚点标签之间的相似度

fprintf('   > 预计算标签的高斯核相似度 R_y (m x m)...\n');

% 存储所有 R_y 矩阵 (Cell array)
% m 很小 (如100)，m*m*L 内存占用极低
R_list = cell(1, num_labels);
H_list = zeros(1, num_labels);

% 并行计算每个标签的自相关矩阵和熵
for l = 1:num_labels
    u_vec = U_labels(:, l);       % m x 1 (软标签: 0.0~1.0)
    y_vec_raw = Y_train(:, l);    % N x 1 (原始0/1标签)

    % [关键] 使用原始数据计算带宽 delta
    % 锚点标签是平滑后的，方差较小。
    % 为了覆盖真实的分布范围，必须用原始样本的 std。
    delta = std(y_vec_raw);
    if delta == 0, delta = 1; end

    % 计算 m x m 距离: (u_j - u_k)^2
    % 利用广播机制计算锚点两两间距离
    diff_mat = abs(u_vec - u_vec');
    dist_sq = diff_mat.^2;

    % 标签模糊相似度 R_y (m x m)
    R_y = exp(-dist_sq / (2 * delta^2));

    R_list{l} = R_y;

    % 计算单标签模糊熵 H(Y) - 基于 m 个锚点
    card = sum(R_y, 2);
    H_list(l) = -mean(log2(card / m_anchors));
end

% ======================================================
% 4. 计算 L x L 互信息矩阵
% ======================================================
LL_SU = zeros(num_labels, num_labels);

fprintf('   > 计算两两标签互信息 (基于 m x m)...\n');

% 外层循环并行加速
for i = 1:num_labels
    R_i = R_list{i}; % m x m
    H_i = H_list(i);

    row_su = zeros(1, num_labels);

    for j = i:num_labels
        if i == j
            row_su(j) = 1;
            continue;
        end

        R_j = R_list{j}; % m x m
        H_j = H_list(j);

        % 联合模糊关系 (Min Operator)
        % 两个 m x m 矩阵取交集，计算量极小
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
    LL_SU(i, :) = row_su;
end

% 补全下三角
for i = 1:num_labels
    for j = i+1:num_labels
        LL_SU(j, i) = LL_SU(i, j);
    end
end

fprintf('全部完成。\n');
end