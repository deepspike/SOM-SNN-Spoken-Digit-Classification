 function AllWeights = GnrlTmptrClsTr_L2(nAfferents,nPtns, nCls, Tmax, ptn, ClassLabels)
% General tempotron classifier for train (Multi-Class separation).
% Inputs:
%       nAfferents--number of input afferents; nPtns--number of trained
%       patterns; nCls--number of classes; Tmax--a vector of nPtns by 1
%       containing the timewindow of each pattern; ptn--is a cell structure,
%       ptn{i,j} contains the spike
%       train of jth afferent from ith pattern. ClassLabels--a row vector of
%       the label of each pattern (1,2...nCls). All the time infromation is
%       in a same scale (1 ms is represented as 1e-3 here).

V_thr_base = 1.0;
V_rest = 0;
margin = 0.5; % adptive threshold
dt = 1e-3;
tau_m = 15e-3;
tau_s = tau_m/4;
V0 = 1/max(exp(-(0:dt:7*tau_m)/tau_m)-exp(-(0:dt:7*tau_m)/tau_s));
lmd = 2e-2/V0;  
maxEpoch = 10;
nNeurons = nCls;
nNeuronPerOutput = 1;
r = 0.1; % constant for reg term
lrDecayRate = 0.99;
AllWeights = 1e-3*randn(nAfferents,nNeurons,nNeuronPerOutput);

for iNeuron=1:nNeurons
    ClassFix = (ClassLabels==iNeuron);
    
    for indNeuronPerOutput = 1:nNeuronPerOutput
        weights = AllWeights(:,iNeuron,indNeuronPerOutput);
        correctRate = zeros(1,maxEpoch);
        
        for epoch=1:maxEpoch
            RandomSqc = randperm(nPtns);  % random perm the training sequence.
            Class = ClassFix(RandomSqc); % instructor
            TTmax = Tmax(RandomSqc);
            Class_Tr = false(1,nPtns); % trained labels
                        
            for pp=1:nPtns
                if Class(pp) == 1
                    V_thr = V_thr_base + margin;
                else
                    V_thr = V_thr_base - margin;
                end
                
                pIdx = RandomSqc(pp);
                Vmax=0; tmax=0;
                fired=false;
                T = TTmax(pp);
                firedAfferents=find(~cellfun('isempty', ptn(pIdx,:)));
                for t=dt:dt:T
                    Vm = 0;
                    
                    if fired == false
                        tSearch = t;
                    else
                        tSearch = t_fire;
                    end
                    
                    for j = firedAfferents
                        
                        Tsyn = find(ptn{pIdx,j}<=tSearch+0.1*dt);    % no cut window
                        
                        if isempty(Tsyn)
                            continue;
                        else
                            sumK =sum(V0*(exp(-(t-ptn{pIdx,j}(Tsyn))/tau_m)-exp(-(t-ptn{pIdx,j}(Tsyn))/tau_s))); % the sum of kernel for No.j afferent
                            Vm = Vm + weights(j)*sumK ;
                        end
                    end 
                  
                    Vm = Vm + V_rest;
                    
                    if Vm >= V_thr && fired==false % fire
                        fired = true;
                        t_fire = t;
                        Class_Tr(pp)=true;
                    end
                    if Vm > Vmax
                        Vmax=Vm; tmax=t;
                    end
                end
                
                if Vmax<=0
                    tmax=T;
                end
                
                if Class_Tr(pp) ~= Class(pp)  % error
                    for k = 1:numel(firedAfferents)
                        j = firedAfferents(k);
                        Tsyn = find(ptn{pIdx,j}<=tmax+0.1*dt);
                        
                        if isempty(Tsyn)
                            continue;
                        else
                            sumK =sum(V0*(exp(-(tmax-ptn{pIdx,j}(Tsyn))/tau_m)-exp(-(tmax-ptn{pIdx,j}(Tsyn))/tau_s))); % the sum of kernel for No.j afferent
                            if fired == false  % LTP
								 Dw = - sumK*2*(V_thr-Vmax);
                                 lr = lmd;
                            else             %LTD 
								 Dw =  sumK*2*(Vmax-V_thr);
                                 lr = lmd;
                            end
                            reg = 2*r*weights(j);
                            
                            % SGD update 
                            update = optim(weights(j),lr,Dw,reg);
                            weights(j) = optim.sgd(update);                           
                        end
                    end
                end
                
            end % end of one iteration (all patterns)
            
            correctRate(epoch) = sum(Class==Class_Tr)/length(Class);
            fprintf('Neuron#: %d indNeuronPerOutput#: %d Iteration#: %d   CorrectRate: %6.3f\n',...
                iNeuron,indNeuronPerOutput, epoch, correctRate(epoch));
            
            if abs( correctRate(epoch)-1 ) < 1e-3
                break;
            end
            lmd = lmd * lrDecayRate; % weight decay after every epoch
        end
        
        AllWeights(:,iNeuron,indNeuronPerOutput) = weights;
        if abs( correctRate(epoch)-1 ) < 1e-3
            fprintf('Succeed\n');
        else
            fprintf('Fail\n');
        end
    end
    
end
end