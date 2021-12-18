function [nl, gradnl] = nmargl_mtgp(logtheta, logtheta_all, covfunc_x, x, y,...
				      N_out, irank, nx, idx_out, idx_in, deriv_range)
                  
% Marginal likelihood and its gradients for multi-output MTGP model
% Modified from https://github.com/ebonilla/mtgp (Copyright (c) 2009, Edwin
% V. Bonilla)
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
% - irank: Rank of Kf 
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
n = length(y);  % Total number of output observations

logtheta_all(deriv_range) = logtheta;  % set the current value of hyperparameters

% Output covariance Kf
n_lf = irank*(2*N_out-irank+1)/2;    % number of parameters of Lf
theta_lf = logtheta_all(1:n_lf);    % parameters of Lf 
Lf = vec2lowtri_inchol(theta_lf,N_out,irank);
Kf = Lf*Lf';

% Input covariance Kx
n_theta_x = eval(feval(covfunc_x{:}));  % number of parameters of Kx
theta_x = logtheta_all(n_lf+1:n_lf+n_theta_x);  % parameters of Kx
Kx = feval(covfunc_x{:}, theta_x, x);

% Noise matrix
sigma2n = exp(2*logtheta_all(n_lf+n_theta_x+1:end-1));  % Noise parameters
Sigma2n = diag(sigma2n);    % Noise Matrix
Var_nx = diag(1./nx);

% Multi-output covariance 
K = Kf(idx_out,idx_out).*Kx(idx_in,idx_in);
K = K + (Sigma2n(idx_out,idx_out).*Var_nx); 
Sigma_noise = MIN_NOISE*eye(n);
K = K + Sigma_noise;

% Alpha
mu = logtheta_all(end);     % constant mean function

L = chol(K)';   % cholesky factorization of the covariance
alpha = solve_chol(L',y-mu);

% Negative log-likelihood
nl = 0.5*(y-mu)'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);
 
% If requested, compute its partial derivatives
if (nargout == 2)   
    
  gradnl = zeros(size(logtheta));  % set the size of the derivative vector
  W = L'\(L\eye(n))-alpha*alpha';   % precompute for convenience
  dmu = 0;
  count = 1;
  
  for zz = 1 : length(deriv_range) 
      
     z = deriv_range(zz);       
     
    if z <= n_lf    % Gradient wrt Kf
      [o,p] = pos2ind_tri_inchol(z,N_out,irank); % determines row and column
      J = zeros(N_out,N_out); J(o,p) = 1;
      Val = J*Lf' + Lf*J';      
      dK = Val(idx_out,idx_out).*Kx(idx_in,idx_in);
      
    elseif z <= (n_lf+n_theta_x)    % Gradient wrt parameters of Kx
      z_x =  z - n_lf;
      dKx = feval(covfunc_x{:},theta_x, x, z_x);      
      dK = Kf(idx_out,idx_out).*dKx(idx_in,idx_in);

    elseif z >= (n_lf+n_theta_x+1) && z <= (n_lf+n_theta_x+N_out)     % Gradient wrt Noise variances
      Val = zeros(N_out,N_out);
      kk = z - n_lf - n_theta_x;
      Val(kk,kk) = 2*Sigma2n(kk,kk);
      dK = Val(idx_out,idx_out).*Var_nx;
    
    elseif z > (n_lf+n_theta_x+N_out)   % Gradient wrt constant mean function
      dmu = -sum(alpha);
        
    end % endif z
    
    if z>n_lf+n_theta_x+N_out
        gradnl(count) = dmu;
    else
        gradnl(count) =  sum(sum(W.*dK,2),1)/2;
    end
    
    count = count + 1;
  end % end for derivarives
  
end % end if nargout ==2





 

