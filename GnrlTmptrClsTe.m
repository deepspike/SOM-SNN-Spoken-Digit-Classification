function classindex = GnrlTmptrClsTe(AllWeights,nPtns, Tmax, ptn)
% General tempotron classifier for test (Multi-Class separation).
% Inputs:
%       AllWeights is nAfferents by nCls, trained weights;
%       nAfferents--number of input afferents; nPtns--number of trained
%       patterns; nCls--number of classes; Tmax--a vector of nPtns by 1
%       containing the timewindow of each pattern; ptn--is a cell structure,
%       ptn{i,j} contains the spike
%       train of jth afferent from ith pattern. ClassLabels--a row vector of
%       the label of each pattern (1,2,3...nCls). All the time infromation is
%       in a same scale (1 ms is represented as 1e-3 here).

[nAfferents,nCls,nNeuronPerOutput] = size(AllWeights);
nNeurons = nCls;
V_thr = 1.5; 
V_rest = 0;
dt = 1e-3;
tau_m = 15e-3;
tau_s = tau_m/4;
V0 = 1/max(exp(-(0:dt:7*tau_m)/tau_m)-exp(-(0:dt:7*tau_m)/tau_s));
neurFires = false(nPtns,nNeurons*nNeuronPerOutput); % Neurons' firing activity
mxVs = zeros(nPtns,nNeurons*nNeuronPerOutput); % Neurons' max voltage
firedAfferents=find(~cellfun('isempty', ptn));
pp=1;
T = Tmax(pp);
tFireCls = ones(nCls,1)*T;
%TmaxRecord = zeros(nNeurons,nPtns);

if nNeuronPerOutput==1
    for iNeuron=1:nCls
        t_fire = T;
        weights = AllWeights(:,iNeuron,1 );
        
        Vmax=0;
        fired=false;
        
        for t=dt:dt:T
            Vm = 0;
            if fired==false
                tSearch=t;
            else
                tSearch=t_fire;
            end
            
            for j=firedAfferents
                Tsyn=find(ptn{pp,j}<=tSearch+0.1*dt);    % no cut window
                
                if isempty(Tsyn)
                    continue;
                else
                    sumK =sum(V0*(exp(-(t-ptn{pp,j}(Tsyn))/tau_m)-exp(-(t-ptn{pp,j}(Tsyn))/tau_s))); % the sum of kernel for No.j afferent
                    Vm = Vm + weights(j)*sumK ;
                end
            end
            Vm = Vm + V_rest;
            if Vm>=V_thr && fired==false % fire
                fired=true;
                t_fire=t;
            end
            if Vm>Vmax
                Vmax=Vm;
            end
        end
        neurFires(pp,iNeuron) = fired;
        mxVs(pp,iNeuron) = Vmax;
        tFireCls(iNeuron) = t_fire;
    end
    
    if ismember(true,neurFires) 
        if nnz(neurFires) == 1
            classindex = find(neurFires == 1);
        else           
            % if exist multiple neuron fire select the earliest fire one
            [tMin,classindex] = min(tFireCls);
            % two neuron fire at the same time, select maximum
            % potential one
            if  numel(find(tFireCls == tMin)) == 2 
                classLabel = find(tFireCls == tMin);
                if mxVs(classLabel(1)) > mxVs(classLabel(2))
                    classindex = classLabel(1);
                else
                    classindex = classLabel(2);
                end
            end
        end
    else  % if no exist neuron fire select the maximum potential one
        pVs = mxVs(pp,:);  %  for displaying the progress
        [~,IdxMxV] = max(pVs);
        t_fire = tFireCls(IdxMxV);
        classindex=IdxMxV;
    end
else    
    for iNeuron=1:nNeurons
        for  indNeuronPerOutput = 1:nNeuronPerOutput
            out = false;
            weights = AllWeights(:,iNeuron,indNeuronPerOutput);
            
            Vmax=0;
            fired=false;
            
            for t=dt:dt:T
                Vm = 0;
                if fired==false
                    tSearch=t;
                else
                    tSearch=t_fire;
                end
                
                for j=firedAfferents
                    Tsyn=find(ptn{pp,j}<=tSearch+0.1*dt);    % no cut window
                    
                    if isempty(Tsyn)
                        continue;
                    else
                        sumK =sum(V0*(exp(-(t-ptn{pp,j}(Tsyn))/tau_m)-exp(-(t-ptn{pp,j}(Tsyn))/tau_s))); % the sum of kernel for No.j afferent
                        Vm = Vm + weights(j)*sumK ;
                    end
                end
                Vm = Vm + V_rest;
                if Vm>=V_thr && fired==false % fire
                    fired=true;
                    t_fire=t;
                    out = fired;
                end
                if Vm>Vmax
                    Vmax=Vm;
                end
            end
            
            indTotNeuronOutput = ((iNeuron-1)*nNeuronPerOutput+indNeuronPerOutput);
            neurFires(pp,indTotNeuronOutput) = out;
            mxVs(pp,indTotNeuronOutput) = Vmax;
        end
    end
    
%     freqs = zeros(nNeurons,1);
%     for neuron = 1:nNeurons
%         for indNeuronPerOutput = 1:nNeuronPerOutput            
%             indTotNeuronOutput = ((neuron-1)*nNeuronPerOutput+indNeuronPerOutput);
%             o = neurFires(1,indTotNeuronOutput);
%             freqs(neuron,1) = freqs(neuron,1) + o;
%         end
%     end
%     
%     freqs1 = freqs(:,1);
%     if sum(freqs1) > 0
%         classindex = find(freqs1==max(freqs1),1);
%     else
    classMxV = zeros(nNeurons,1);
    
    for iNeuron=1:nNeurons
        for  indNeuronPerOutput = 1:nNeuronPerOutput
            classMxV(iNeuron) = classMxV(iNeuron)+mxVs((iNeuron-1)*nNeuronPerOutput+indNeuronPerOutput);
        end
    end

    [maxV,classindex] = max(classMxV);
    end
end

