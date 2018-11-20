clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Mt = 64;
Mr = Mt;
snr_db = [30]; % the transmit signal-to-noise ratio
snr = 10^(-snr_db/10);
total_num_of_clusters = 3;
total_num_of_rays = 10;
L = total_num_of_clusters*total_num_of_rays; % Total number of distinct paths of the mmWave channel
Imax = 100; % maximum number of iterations for the iterative algorithms
T_range = [400 800 2458];
maxMCRealizations = 1;

%% Variables initialization
mean_error_svt = zeros(length(T_range), Imax);
mean_error_proposed = zeros(length(T_range), Imax);

Dr = 1/sqrt(Mr)*exp(-1j*[0:Mr-1]'*2*pi*[0:Mr-1]/Mr);
Dt = 1/sqrt(Mt)*exp(-1j*[0:Mt-1]'*2*pi*[0:Mt-1]/Mt);
B = kron(conj(Dt), Dr);

for r=1:maxMCRealizations
  disp(['realization: ', num2str(r)]);
  convergence_error_svt = zeros(length(T_range), Imax);  
  convergence_error_mc = zeros(length(T_range), Imax);  
  convergence_error_proposed = zeros(length(T_range), Imax);

  %%% Signal formulation (channel and training sequence)
  
  for sub_indx=1:length(T_range)

    % Create the mmWave MIMO channel
    [H,Ar,At] = generate_mmwave_channel(Mr, Mt, total_num_of_clusters, total_num_of_rays);
    
    % Get the measurements at the RX of the transmitted training symbols   
    [y,M,OH,Omega] = get_measurements_at_RX(H, T, snr, B);

    % SVT matrix completion
    [~, convergence_error_svt(sub_indx, :)] = mc_svt(H, OH, Omega, Imax);
    
    % Proposed technique based on ADMM matrix completion with side-information
    rho = 0.005;
    tau_S = .1/(1+snr_db);
    [~, convergence_error_proposed(sub_indx, :)] = proposed_algorithm(H, OH, Omega, Fr, Ft, Imax, rho*norm(OH), tau_S, rho, 1);

  end
  mean_error_svt = mean_error_svt + convergence_error_svt;
  mean_error_proposed = mean_error_proposed + convergence_error_proposed;
end
mean_error_svt = mean_error_svt/maxMCRealizations;
mean_error_proposed = mean_error_proposed/maxMCRealizations;

%% Plotting
figure;
marker_stepsize = 50;
p_svt_1 = semilogy(1:Imax,  (mean_error_svt(1, :)));hold on;
set(p_svt_1,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p_svt_2 = semilogy(1:Imax,  (mean_error_svt(2, :)));hold on;
set(p_svt_2,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 8, 'Color', 'Blue');
p_svt_3 = semilogy(1:Imax,  (mean_error_svt(3, :)));hold on;
set(p_svt_3,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'p', 'MarkerSize', 8, 'Color', 'Green');

p_mcsi_1 = semilogy(1:Imax,  (mean_error_proposed(1, :)));hold on;
set(p_mcsi_1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p_mcsi_2 = semilogy(1:Imax,  (mean_error_proposed(2, :)));hold on;
set(p_mcsi_2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 8, 'Color', 'Blue');
p_mcsi_3 = semilogy(1:Imax,  (mean_error_proposed(3, :)));hold on;
set(p_mcsi_3,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'p', 'MarkerSize', 8, 'Color', 'Green');
xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
legend({'T=100','T=200','T=300'}, 'FontSize', 12);
grid on;set(gca,'FontSize',12);

savefig(strcat('results/nmse_admm_iterations_',num2str(Mt), '_',num2str(T_range),'.fig'))