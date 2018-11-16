clear;
clc;

%%% Initialization
Mt = 64;
Mr = Mt;
Gt = Mt;
Gr = Mr;
snr_db = [30];
snr = 10^(-snr_db/10);
subSamplingRatio_range = [0.5];
paths_range = [2:4:24];
Imax = 100;
maxRealizations = 100;

error_mcsi = zeros(maxRealizations,1);
error_omp = zeros(maxRealizations,1);
error_vamp = zeros(maxRealizations,1);
error_twostage = zeros(maxRealizations,1);

mean_error_mcsi = zeros(length(subSamplingRatio_range), length(paths_range));
mean_error_omp =  zeros(length(subSamplingRatio_range), length(paths_range));
mean_error_vamp =  zeros(length(subSamplingRatio_range), length(paths_range));
mean_error_twostage =  zeros(length(subSamplingRatio_range), length(paths_range));

for path_indx = 1:length(paths_range)
  Np = paths_range(path_indx);  

  for sub_indx=1:length(subSamplingRatio_range)

   parfor r=1:maxRealizations
   disp(['realization: ', num2str(r)]);

    [H,Ar,At] = parametric_mmwave_channel(Mr, Mt, 1, Np);
    Fr = 1/sqrt(Mr)*exp(-1j*[0:Mr-1]'*2*pi*[0:Gr-1]/Gr);
    Ft = 1/sqrt(Mt)*exp(-1j*[0:Mt-1]'*2*pi*[0:Gt-1]/Gt);
    S = Fr'*H*Ft;
    [y,M,OH,Omega] = system_model(H, Fr, Ft, round(subSamplingRatio_range(sub_indx)*Mt*Mr), snr);

    % Sparse channel estimation
    s_omp = OMP(M, y, 2*Np);
    X_omp = Fr*reshape(s_omp, Gr, Gt)*Ft';
    S_omp = reshape(s_omp, Gr, Gt);
    error_omp(r) = norm(H-X_omp)^2/norm(H)^2;      

    % ADMM matrix completion with side-information
    rho = 0.005;
    tau_S = .1/(1+snr_db);
    X_mcsi = mcsi_admm(H, OH, Omega, Fr, Ft, Imax, rho*norm(OH), tau_S, rho, 1);
    error_mcsi(r) = norm(H-X_mcsi)^2/norm(H)^2;

    % VAMP sparse recovery
    s_vamp = vamp(y, M, snr, 2*Np);
    X_vamp = Fr*reshape(s_vamp, Gr, Gt)*Ft';
    error_vamp(r) = norm(H-X_vamp)^2/norm(H)^2;


    % Two-stage scheme matrix completion and sparse recovery
    X_twostage_1 = mc_svt(H, OH, Omega, Imax);
    s_twostage = vamp(vec(X_twostage_1), kron(conj(Ft), Fr), 0.001, 2*Np);
    X_twostage = Fr*reshape(s_twostage, Gr, Gt)*Ft';
    error_twostage(r) = norm(H-X_twostage)^2/norm(H)^2;      
   end

    mean_error_mcsi(sub_indx, path_indx) = min(mean(error_mcsi), 1);
    mean_error_omp(sub_indx, path_indx) = min(mean(error_omp), 1);
    mean_error_vamp(sub_indx, path_indx) = min(mean(error_vamp), 1);
    mean_error_twostage(sub_indx, path_indx) = min(mean(error_twostage), 1);

  end

end


figure;
p11 = semilogy(paths_range, (mean_error_omp(1, :)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p12 = semilogy(paths_range, (mean_error_vamp(1, :)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Blue');
p13 = semilogy(paths_range, (mean_error_twostage(1, :)));hold on;
set(p13,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Cyan', 'MarkerFaceColor', 'Cyan', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Cyan');
p14 = semilogy(paths_range, (mean_error_mcsi(1, :)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Green');

legend({'OMP [4]', 'VAMP [12]', 'TSSR [9]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');


xlabel('number of paths N_p');
ylabel('MSE (dB)')
grid on;set(gca,'FontSize',12);

savefig(['results/errorVSrays_',num2str(Mt),'.fig'])