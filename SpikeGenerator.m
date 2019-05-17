function [spikeTrainList,Tmax] = SpikeGenerator(input,dt,nAfferents)
    total_sample = length(input);
    Tmax = zeros(total_sample,1);
    spikeTrainList = cell(total_sample,nAfferents);
    
    for i = 1:total_sample
        sample = input{i,1};
        timeTotal = size(sample,2);
        Tmax(i) = timeTotal*dt;
        for j = 1:timeTotal
            afferent = sample(:,j);
            for k = 1:numel(afferent)
                spikeTrainList{i,afferent(k)} = [spikeTrainList{i,afferent(k)} j*dt];
            end
        end
    end
end