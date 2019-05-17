function spikeTrain = latencyCode(dataCell,max,min)
    [numClass,numSample] = size(dataCell);
    spikeTrain = cell(numClass,numSample);
    
    for iC = 1:numClass
        for idx = 1:numSample
            sample = dataCell{iC,idx}; 
            % x = [min(sample,[],1);max(sample,[],1)];
            % spikeSample = bsxfun(@minus,sample,x(1,:));
            % spikeSample = bsxfun(@rdivide,spikeSample,diff(x,1,1));
            spikeSample =  bsxfun(@minus,sample,min);
            spikeSample =  bsxfun(@rdivide,spikeSample,(max-min));
            spikeTrain{iC,idx} = 1 - spikeSample; % strongest signal fire earliest
        end
    end
end