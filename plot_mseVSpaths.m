clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Mt = 64;
Mr = Mt;
snr_db = [30]; % the transmit signal-to-noise ratio
snr = 10^(-snr_db/10);
T_range = [400]; % training length
total_num_of_clusters = 2; % number of clusters for the mmWave channel
total_num_of_rays = 1; % number of rays for the mmWave channel
L = total_num_of_clusters*total_num_of_rays; % Total number of distinct paths of the mmWave channel
Imax = 100; % maximum number of iterations for the iterative algorithms
paths_range = [2:4:24];
maxMCRealizations = 1;

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_omp = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
error_twostage = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(T_range), length(paths_range));
mean_error_omp =  zeros(length(T_range), length(paths_range));
mean_error_vamp =  zeros(length(T_range), length(paths_range));
mean_error_twostage =  zeros(length(T_range), length(paths_range));

Dr = 1/sqrt(Mr)*exp(-1j*[0:Mr-1]'*2*pi*[0:Mr-1]/Mr);
Dt = 1/sqrt(Mt)*exp(-1j*[0:Mt-1]'*2*pi*[0:Mt-1]/Mt);
B = kron(conj(Dt), Dr);

%% Iterations for different SNRs, training length and MC realizations
for path_indx = 1:length(paths_range)
  Np = paths_range(path_indx);

  for sub_indx=1:length(T_range)
   T = T_range(sub_indx);
   
   for r=1:maxMCRealizations
   disp(['realization: ', num2str(r)]);

    % Create the mmWave MIMO channel
    [H,Ar,At] = generate_mmwave_channel(Mr, Mt, 1, Np);
    
    % Get the measurements at the RX of the transmitted training symbols   
    [y,M,OH,Omega] = get_measurements_at_RX(H, T, snr, B);

    % Sparse channel estimation
    s_omp = OMP(M, y, 2*Np);
    X_omp = Fr*reshape(s_omp, Mr, Mt)*Ft';
    S_omp = reshape(s_omp, Mr, Mt);
    error_omp(r) = norm(H-X_omp)^2/norm(H)^2;      

    % VAMP sparse recovery
    s_vamp = vamp(y, M, snr, 2*Np);
    X_vamp = Fr*reshape(s_vamp, Mr, Mt)*Ft';
    error_vamp(r) = norm(H-X_vamp)^2/norm(H)^2;


    % Two-stage scheme matrix completion and sparse recovery
    X_twostage_1 = mc_svt(H, OH, Omega, Imax);
    s_twostage = vamp(vec(X_twostage_1), kron(conj(Ft), Fr), 0.001, 2*Np);
    X_twostage = Fr*reshape(s_twostage, Mr, Mt)*Ft';
    error_twostage(r) = norm(H-X_twostage)^2/norm(H)^2;
    
    % Proposed technique based on ADMM matrix completion with side-information
    rho = 0.005;
    tau_S = .1/(1+snr_db);
    X_proposed = proposed_algorithm(H, OH, Omega, Fr, Ft, Imax, rho*norm(OH), tau_S, rho, 1);
    error_proposed(r) = norm(H-X_proposed)^2/norm(H)^2;

   end

   mean_error_proposed(sub_indx, snr_indx) = mean(error_proposed);
   mean_error_omp(sub_indx, snr_indx) = mean(error_omp);
   mean_error_vamp(sub_indx, snr_indx) = mean(error_vamp);
   mean_error_twostage(sub_indx, snr_indx) = mean(error_twostage);

  end

end


figure;
p11 = semilogy(paths_range, (mean_error_omp(1, :)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p12 = semilogy(paths_range, (mean_error_vamp(1, :)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Black', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p13 = semilogy(paths_range, (mean_error_twostage(1, :)));hold on;
set(p13,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
p14 = semilogy(paths_range, (mean_error_mcsi(1, :)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Green');

legend({'OMP [4]', 'VAMP [12]', 'TSSR [9]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');


xlabel('number of paths N_p');
ylabel('MSE (dB)')
grid on;set(gca,'FontSize',12);

savefig(['results/errorVSrays_',num2str(Mt),'.fig'])