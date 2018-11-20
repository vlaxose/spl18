clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Mt = 64;
Mr = Mt;
total_num_of_clusters = 2; % number of clusters for the mmWave channel
total_num_of_rays = 1; % number of rays for the mmWave channel
L = total_num_of_clusters*total_num_of_rays; % Total number of distinct paths of the mmWave channel
snr_range = [0:5:25]; % range of the transmit signal-to-noise ratio
T_range = [800]; % training length
Imax = 50; % maximum number of iterations for the iterative algorithms
maxMCRealizations = 1;

%% Variables initialization

rate_proposed = zeros(maxMCRealizations,1);
rate_opt = zeros(maxMCRealizations,1);
rate_omp = zeros(maxMCRealizations,1);
rate_vamp = zeros(maxMCRealizations,1);
rate_twostage = zeros(maxMCRealizations,1);
mean_rate_proposed = zeros(length(T_range), length(snr_range));
mean_rate_opt =  zeros(length(T_range), length(snr_range));
mean_rate_omp =  zeros(length(T_range), length(snr_range));
mean_rate_vamp =  zeros(length(T_range), length(snr_range));
mean_rate_twostage =  zeros(length(T_range), length(snr_range));

Dr = 1/sqrt(Mr)*exp(-1j*[0:Mr-1]'*2*pi*[0:Mr-1]/Mr);
Dt = 1/sqrt(Mt)*exp(-1j*[0:Mt-1]'*2*pi*[0:Mt-1]/Mt);
B = kron(conj(Dt), Dr);


%% Iterations for different SNRs, training length and MC realizations
for snr_indx = 1:length(snr_range)
  snr = 10^(-snr_range(snr_indx)/10);

  for sub_indx=1:length(T_range)
   T = T_range(sub_indx);
   
   for r=1:maxMCRealizations
   disp(['realization: ', num2str(r)]);

    % Create the mmWave MIMO channel
    [H,Ar,At] = generate_mmwave_channel(Mr, Mt, total_num_of_clusters, total_num_of_rays);
    
    % Get the measurements at the RX of the transmitted training symbols   
    [y,M,OH,Omega] = get_measurements_at_RX(H, T, snr, B);

    [Uh,Sh,Vh] = svd(H);
    rate_opt(r) = log2(real(det(eye(Mr)+1/(Mt*Mr)*1/snr*H*H')));


    % VAMP sparse recovery
    s_vamp = vamp(y, M, snr, 2*L);
    X_vamp = Dr*reshape(s_vamp, Mr, Mt)*Dt';
    [U_vamp, S_vamp, V_vamp] = svd(X_vamp);
    rate_vamp(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_vamp)^2/norm(H)^2))));

    % Sparse channel estimation
    s_omp = OMP(M, y, 2*L);
    X_omp = Dr*reshape(s_omp, Mr, Mt)*Dt';
    [U_omp, S_omp, V_omp] = svd(X_omp);
    rate_omp(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_omp)^2/norm(H)^2))));
    
    % Proposed technique based on ADMM matrix completion with side-information
    rho = 0.005;
    tau_S = .1/(1+snr_range(snr_indx));
    X_mcsi = proposed_algorithm(H, OH, Omega, Dr, Dt, Imax, rho*norm(OH), tau_S, rho, 1);
    [U_mcsi,S_mcsi,V_mcsi] = svd(X_mcsi);
    rate_proposed(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_mcsi)^2/norm(H)^2))));
    
    % Two-stage scheme matrix completion and sparse recovery
    X_twostage_1 = mc_svt(H, OH, Omega, Imax);
    s_twostage = vamp(vec(X_twostage_1), kron(conj(Dt), Dr), 0.001, 2*L);
    X_twostage = Dr*reshape(s_twostage, Mr, Mt)*Dt';
    [U_twostage,S_twostage,V_twostage] = svd(X_twostage);
    rate_twostage(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_twostage)^2/norm(H)^2))));
   end

    mean_rate_proposed(sub_indx, snr_indx) = mean(rate_proposed);
    mean_rate_omp(sub_indx, snr_indx) = mean(rate_omp);
    mean_rate_opt(sub_indx, snr_indx) = mean(rate_opt);
    mean_rate_vamp(sub_indx, snr_indx) = mean(rate_vamp);
    mean_rate_twostage(sub_indx, snr_indx) = mean(rate_twostage);

  end

end


figure;
p11 = plot(snr_range, (mean_rate_omp(1, :)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p12 = plot(snr_range, (mean_rate_twostage(1, :)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Black');
p15 = plot(snr_range, (mean_rate_vamp(1, :)));hold on;
set(p15,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'o', 'MarkerSize', 8, 'Color', 'Blue');
p14 = plot(snr_range, (mean_rate_proposed(1, :)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Green');
p16 = plot(snr_range, (mean_rate_opt(1, :)));hold on;
set(p16,'LineWidth',2, 'LineStyle', '--', 'Color', 'Black');

legend({'OMP [4]', 'TSSR [9]', 'VAMP [12]', 'Proposed', 'Perfect CSI'}, 'FontSize', 12);

xlabel('SNR (dB)');
ylabel('ASE')
grid on;set(gca,'FontSize',12);

savefig(strcat('results/rateVSsnr.fig'))