function [nl, gradnl] = nmargl_slfm(logtheta, logtheta_all, covfunc_x, x, y,...
    N_out, Q, nx, idx_out, idx_in, deriv_range)

% Marginal likelihood and its gradients for multi-output SLFM model
%
% nl = nmargl_mtgp(logtheta, ...) Returns the negative log marginal likelihood
% [nl gradnl] =  nmargl_mtgp(logtheta, ...) Returns also the gradients wrt logtheta
%
% INPUT
% - logtheta: Column vector of current values of hyperparameters to be optimized
% - logtheta_all: Vector of all parameters: [theta_lf; theta_x; sigma_l; mu]
%                - theta_lf: the parameter vector of the
%                   cholesky decomposition of Kf
%                - theta_x: the parameters of Kx
%                - sigma_l: The log of the noise std deviations for each task
%                - mu: the constant mean function 
% - covfunc_x: Name of covariance function on input space x
% - x: Unique input points
% - y: Vector of output values
% - N_out: Number of outputs
% - Q: number of latent functions 
% - nx: number of times each element of y has been observed 
%                usually nx(i)=1 unless the corresponding y is an average
% - idx_out: Vector containing the indexes of the output to which
%                each observation y corresponds
% - idx_in: Vector containing the indexes of the x data-points to
%                which each observation y corresponds
% - deriv_range: The indices of the parameters in logtheta_all
%                to which each element in logtheta corresponds


% *** General settings here ****
MIN_NOISE = 0;
% ******************************

if ischar(covfunc_x), covfunc_x = cellstr(covfunc_x); end % convert to cell if needed

D = size(x,2);  % Dimensionality used when covfunc_x is called 
n = length(y); % Total number of output observations

logtheta_all(deriv_range) = logtheta;

% Build the multi-output covariance 
K = zeros(n,n);
ltheta_x = [];
theta_x = {};
Kf_q = {};
Kx_q = {};
theta_lf = {};
cc_x = 0;
cc_lf = 0;
nlf = N_out;

for iq = 1:Q    % for each latent function 
    
    theta_lf{iq} = logtheta_all(cc_lf+1:cc_lf+nlf);
    cc_lf = cc_lf + nlf;
    Kf_q{iq} = theta_lf{iq}*theta_lf{iq}';  % Output covariance matrix 
    
    ltheta_x(iq) = eval(feval(covfunc_x{iq}{:}));
    theta_x{iq} = logtheta_all(N_out*Q+cc_x+1:N_out*Q+cc_x+ltheta_x(iq));
    cc_x = cc_x + ltheta_x(iq);
    Kx_q{iq} = feval(covfunc_x{iq}{:}, theta_x{iq}, x); % Input covariance matrix 
    
    K = K + Kf_q{iq}(idx_out,idx_out).*Kx_q{iq}(idx_in,idx_in);
end

% Noise matrix                  
sigma2n = exp(2*logtheta_all(cc_lf+cc_x+1:end-1));  % Noise parameters
Sigma2n = diag(sigma2n);    % Noise matrix
Var_nx = diag(1./nx);

K = K + (Sigma2n(idx_out,idx_out).*Var_nx);
Sigma_noise = MIN_NOISE*eye(n);
K = K + Sigma_noise;

% Alpha
mu = logtheta_all(end);     % constant mean function

L = chol(K)';       % cholesky factorization of the covariance
alpha = solve_chol(L',y-mu);

% Negative log-likelihood
nl = 0.5*(y-mu)'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);

% If requested, compute its partial derivatives
if (nargout == 2)
    
    gradnl = zeros(size(logtheta));        % set the size of the derivative vector
    W = L'\(L\eye(n))-alpha*alpha';      % precompute for convenience
    dmu = 0;
    count = 1;
    
    for zz = 1 : length(deriv_range)
        
        z = deriv_range(zz);
        
        if z <= cc_lf    % Gradient wrt Kf
            iq = ceil(z/N_out);
            z = z-(iq-1)*N_out;
            J = zeros(N_out,1); J(z) = 1;
            Val = J* theta_lf{iq}' +  theta_lf{iq}*J'; 
            dK = Val(idx_out,idx_out).*Kx_q{iq}(idx_in,idx_in);
            
        elseif z <= (cc_lf+cc_x)    % Gradient wrt parameters of Kx
            z_x =  z - cc_lf;
            iq = find(cumsum([ltheta_x(1) ltheta_x])>=z_x,1);
            if isempty(iq) iq = Q; end
            if iq~=1
                z_x = z_x - sum(ltheta_x(1:iq-1));
            end
            
            dKx = feval(covfunc_x{iq}{:},theta_x{iq}, x, z_x);
            dK = Kf_q{iq}(idx_out,idx_out).*dKx(idx_in,idx_in);
                    
        elseif z >= (cc_lf+cc_x+1) && z <= (cc_lf+cc_x+N_out)    % Gradient wrt Noise variances
            Val = zeros(N_out,N_out);
            kk = z - cc_lf - cc_x;
            Val(kk,kk) = 2*Sigma2n(kk,kk);
            dK = Val(idx_out,idx_out).*Var_nx;
            
        elseif z > (cc_lf+cc_x+N_out)   % Gradient wrt constant mean function
            dmu = -sum(alpha);
            
        end % endif z
        
        if z>cc_lf+cc_x+N_out
            gradnl(count) = dmu;
        else
            gradnl(count) =  sum(sum(W.*dK,2),1)/2;
        end
        
        count = count + 1;
    end % end for derivarives
    
end % end if nargout ==2







