
function [K_X, K_XY, K_Y] = PSD_ALL(K_X_in, K_XY_in, K_Y_in)
% PSD_ALL:
V = numel(K_X_in);

% S_Y -> K_Y
K_Y = make_psd(K_Y_in);

%% ================= 3. Compute Label Kernel Square Roots =================

% K_Y^(1/2) and K_Y^dagger/2
[K_Y_half, K_Y_dag_half] = psd_sqrt_pair(K_Y);


%% ================= 4. Process Each View =================

K_X = cell(1, V);
K_XY = cell(1, V);

for v = 1:V

    % Raw dependency matrices
    S_Xv  = K_X_in{v};
    S_XYv = K_XY_in{v};

    %% 4.1 PSD correction of feature kernel

    % S_X^v -> K_X^v
    K_Xv = make_psd(S_Xv);


    %% 4.2 Compute feature kernel square roots

    % K_X^(1/2) and K_X^dagger/2
    [K_Xv_half, K_Xv_dag_half] = psd_sqrt_pair(K_Xv);


    %% 4.3 Normalize cross-space dependency matrix

    % R_0^v = K_X^dagger/2 * S_XY * K_Y^dagger/2
    R0 = K_Xv_dag_half * S_XYv * K_Y_dag_half;


    %% 4.4 Projection onto spectral norm unit ball

    % SVD: R0 = U * Sigma * V'
    [U, Sigma, V_svd] = svd(R0, 'econ');

    singular_values = diag(Sigma);

    % Singular value clipping:
    % sigma_i -> min(sigma_i, 1)
    singular_values = min(singular_values, 1);

    % Reconstruct normalized cross-dependency matrix
    R = U * diag(singular_values) * V_svd';
    %% 4.5 Reconstruct cross-space kernel

    % K_XY = K_X^(1/2) * R * K_Y^(1/2)
    K_XYv = K_Xv_half * R * K_Y_half;
    % Remove numerical errors
    K_XYv = real(K_XYv);

    %% 4.6 Save results

    K_X{v} = K_Xv;
    K_XY{v} = K_XYv;

end

end


% ==========================================================
% Function 1: PSD correction by eigenvalue truncation
% ==========================================================

function K_psd = make_psd(K)
    % Symmetrization
    K = (K + K') / 2;
    % Eigenvalue decomposition
    [U, D] = eig(K);
    eigenvalues = real(diag(D));
    % Truncate negative eigenvalues to zero
    eigenvalues = max(eigenvalues, 0);
    % Reconstruct PSD matrix
    K_psd = U * diag(eigenvalues) * U';
    % Numerical symmetrization
    K_psd = real(K_psd);
    K_psd = (K_psd + K_psd') / 2;

end


% ==========================================================
% Function 2: PSD matrix square root and pseudoinverse
% ==========================================================

function [K_half, K_dag_half] = psd_sqrt_pair(K)

    % Eigenvalue decomposition
    K = (K + K') / 2;

    [U, D] = eig(K);

    eigenvalues = real(diag(D));

    % Remove tiny negative numerical errors
    eigenvalues = max(eigenvalues, 0);

    % Numerical rank tolerance
    max_eig = max(eigenvalues);

    if max_eig > 0
        tol = max(size(K)) * eps(max_eig);
    else
        tol = 0;
    end

    % Ordinary square root
    sqrt_eigenvalues = sqrt(eigenvalues);

    % Pseudoinverse square root
    inv_sqrt_eigenvalues = zeros(size(eigenvalues));

    valid_idx = eigenvalues > tol;

    inv_sqrt_eigenvalues(valid_idx) = ...
        1 ./ sqrt(eigenvalues(valid_idx));

    % Reconstruct both matrices
    K_half = U * diag(sqrt_eigenvalues) * U';

    K_dag_half = U * diag(inv_sqrt_eigenvalues) * U';

    % Numerical correction
    K_half = real((K_half + K_half') / 2);

    K_dag_half = real((K_dag_half + K_dag_half') / 2);

end
