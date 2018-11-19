clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Mt = 64; % number of TX antennas
Mr = Mt; % number of RX antennas
T_range = [400]; % training length
total_num_of_clusters = 2; % number of clusters for the mmWave channel
total_num_of_rays = 1; % number of rays for the mmWave channel
L = total_num_of_clusters*total_num_of_rays; % Total number of distinct paths of the mmWave channel
snr_range = [0:5:25]; % range of the transmit signal-to-noise ratio
Imax = 100; % maximum number of iterations for the iterative algorithms
maxMCRealizations = 50;

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_omp = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
error_twostage = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(T), length(snr_range));
mean_error_omp =  zeros(length(T), length(snr_range));
mean_error_vamp =  zeros(length(T), length(snr_range));
mean_error_twostage =  zeros(length(T), length(snr_range));

Dr = 1/sqrt(Mr)*exp(-1j*[0:Mr-1]'*2*pi*[0:Mr-1]/Mr);
Dt = 1/sqrt(Mt)*exp(-1j*[0:Mt-1]'*2*pi*[0:Mt-1]/Mt);
B = kron(conj(Dt), Dr);

%% Iterations for different SNRs, , training length and MC realizations
for snr_indx = 1:length(snr_range)
  snr = 10^(-snr_range(snr_indx)/10);

  for sub_indx=1:length(T_range)
   T = T_range(sub_indx);
    
   parfor r=1:maxMCRealizations
    disp(['=> SNR:', num2str(snr_range(snr_indx)), 'dB, ', 'realization: ', num2str(r)]);

    % Create the mmWave MIMO channel
    [H,Ar,At] = generate_mmwave_channel(Mr, Mt, total_num_of_clusters, total_num_of_rays);
    
    % Get the measurements at the RX of the transmitted training symbols   
    [y,M,OH,Omega] = get_measurements_at_RX(H, T, snr, B);

    % Two-stage scheme matrix completion and sparse recovery
    disp('Running Two-stage-based Technique for low-rank and sparse reconstruction..');
    X_twostage_1 = mc_svt(H, OH, Omega, Imax);
    s_twostage = vamp(vec(X_twostage_1), kron(conj(Dt), Dr), 0.001, 2*L);
    X_twostage = Dr*reshape(s_twostage, Mr, Mt)*Dt';
    error_twostage(r) = norm(H-X_twostage)^2/norm(H)^2;
    
    % VAMP sparse recovery
    disp('Running VAMP-based sparse reconstruction...');
    s_vamp = vamp(y, M, snr, 2*L);
    X_vamp = Dr*reshape(s_vamp, Mr, Mt)*Dt';
    error_vamp(r) = norm(H-X_vamp)^2/norm(H)^2;
    
    % Sparse channel estimation
    disp('Running OMP-based sparse reconstruction...');
    s_omp = OMP(M, y, 2*L);
    X_omp = Dr*reshape(s_omp, Mr, Mt)*Dt';
    S_omp = reshape(s_omp, Mr, Mt);
    error_omp(r) = norm(H-X_omp)^2/norm(H)^2;      

    % Proposed technique based on ADMM matrix completion with side-information
    disp('Running proposed algorithm...');
    rho = 0.005;
    tau_S = .1/(1+snr_range(snr_indx));
    X_proposed = proposed_algorithm(H, OH, Omega, Dr, Dt, Imax, rho*norm(OH), tau_S, rho, 1);
    error_proposed(r) = norm(H-X_proposed)^2/norm(H)^2;

   end

    mean_error_proposed(sub_indx, snr_indx) = mean(error_proposed);
    mean_error_omp(sub_indx, snr_indx) = mean(error_omp);
    mean_error_vamp(sub_indx, snr_indx) = mean(error_vamp);
    mean_error_twostage(sub_indx, snr_indx) = mean(error_twostage);

  end

end


figure;
p11 = semilogy(snr_range, (mean_error_omp(1, :)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p12 = semilogy(snr_range, (mean_error_vamp(1, :)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'o', 'MarkerSize', 8, 'Color', 'Black');
p13 = semilogy(snr_range, (mean_error_twostage(1, :)));hold on;
set(p13,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Black');
p14 = semilogy(snr_range, (mean_error_proposed(1, :)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Green');

legend({'OMP [4]', 'VAMP [12]', 'TSSR [9]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');

xlabel('SNR (dB)');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',12);

savefig(strcat('results/mseVSsnr_',num2str(T),'.fig'))