function label = somTest(input,net,K)
     numSample = size(input,2);
     weight = cell2mat(net.iw);
     label = zeros(K,numSample);
     for i = 1:numSample
         dist = pdist2(input(:,i)',weight);
         [~,I] = sort(dist); % find the index of winning neuron
         label(:,i) = I(1:K);
     end
end
