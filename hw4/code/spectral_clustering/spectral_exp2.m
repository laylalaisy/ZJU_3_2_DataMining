load('TDT2_data', 'fea', 'gnd');

% YOUR CODE HERE
options = [];
options.NeighborMode = 'Supervised';
options.gnd = gnd;
options.bLDA = 1;
W = full(constructW(fea,options));  

k = size(unique(gnd),1);
N = size(gnd, 1);
P = size(fea, 1);

iter = 1;
acc_spectral = 0;
acc_kmeans = 0;
nmi_spectral = 0;
nmi_kmeans = 0;

for i =1:iter
    idx_spectral = spectral(W, k);
    gnd_map_spectral = bestMap(gnd, idx_spectral);    % map idx to gnd label
    acc_spectral = acc_spectral + length(find(gnd == gnd_map_spectral))/N;
    nmi_spectral = nmi_spectral + MutualInfo(gnd, gnd_map_spectral);
    
    idx_kmeans=litekmeans(full(fea), k);
    gnd_map_kmeans = bestMap(gnd, idx_kmeans);
    acc_kmeans = acc_kmeans + length(find(gnd == gnd_map_kmeans))/N;
    nmi_kmeans = nmi_kmeans + MutualInfo(gnd, gnd_map_kmeans);
end

acc_spectral = acc_spectral/iter;
nmi_spectral = nmi_spectral/iter;
acc_kmeans = acc_kmeans/iter;
nmi_kmeans = nmi_kmeans/iter;


fprintf('accuracy of spectral: %f, mutual info of spectral: %f\n', acc_spectral, nmi_spectral);
fprintf('accuracy of kmeans: %f, mutual info of kmeans: %f\n', acc_kmeans, nmi_kmeans);