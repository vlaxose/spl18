function [X, convergence_error] = mcsi_admm(Htrue, OH, Omega, Dr, Dt, Imax, tau, tau_s, rho, flag)

  [Mr, Mt] = size(OH);
  Gr = size(Dr, 2);
  Gt = size(Dt, 2);
  convergence_error = zeros(Imax,1);
  
  Y = zeros(Mr, Mt);
  Z1 = zeros(Mr, Mt);
  Z2 = zeros(Mr, Mt);
  s = zeros(Gr*Gt, 1);
  C = zeros(Gr, Gt);


  A = zeros(Mr*Mt);
  for i=1:Mr
    Eii = zeros(Mr);
    Eii(i,i) = 1;
    A = A + kron(diag(Omega(i, :))', Eii);
  end
  B = kron(conj(Dt), Dr);

  for i=1:Imax

    M = A+2*rho*eye(Mr*Mt);
  
    % sub 1
    X = svt(Y-1/rho*Z1, tau/rho);
    
    % sub 2    
    y = M\(vec(Z1) + rho*vec(X) + vec(OH) + vec(Z2) + rho*vec(C) + rho*vec(B*s));
    Y = reshape(y, Gr, Gt);
    
    % sub 3
    v = B'*vec(Y -C - Z2/rho);
    s = max(abs(real(v))-tau_s/(rho),0).*sign(real(v)) +1j* max(abs(imag(v))-tau_s/(rho),0).*sign(imag(v));
    S = reshape(s, Mr, Mt);
    Xs = Dr*S*Dt';
    
    % sub 4    
    C = rho/(rho+1)*(Y - Xs - Z2/rho);
     
    % dual update
    Z1 = Z1 + rho*(X-Y);
    Z2 = Z2 + rho*(C - Y + Xs);
    
    if(flag)
      rho = 1.02*rho;
    end
    
    convergence_error(i) = norm(X-Htrue)^2/norm(Htrue)^2;

  end
  

end
