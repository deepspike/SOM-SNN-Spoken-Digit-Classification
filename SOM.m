function net = SOM(featureCell,dim1,dim2)
%% Self-Organizing Map Training
numSample = numel(featureCell);
inputs = cell2mat(reshape(featureCell,1,numSample));

% Define the specifications of Self-Organizing Map
dimension1 = dim1;
dimension2 = dim2;
net = selforgmap([dimension1 dimension2],'initNeighbor',11,'coverSteps',200,'distanceFcn','mandist','topologyFcn','gridtop');

% Train the Network
net.trainParam.epochs = 400;
[net,~] = train(net,inputs);

end
