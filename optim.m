classdef optim
% class for weight optimization

    properties
    % define the properties of the class here
        weightOld;
        lr = 0.01;
        dw;
        reg = 0.1;
        param = struct;
    end
    
    methods
        % constructor
        function obj = optim(weightOld,lr,dw,reg)
                obj.weightOld = weightOld;
                obj.lr = lr;  % learning rate
                obj.dw = dw;  % gradient
                obj.reg = reg; % regularization rate
        end        
    end
    
    methods(Static = true)
        function weightNew = sgd(obj)
        % Performs vanilla stochastic gradient descent.
        
            weightNew = obj.weightOld - obj.lr * obj.dw - obj.lr * obj.reg;
        end
        
        function [weightNew,v] = momentum(obj,mu,v)
        % Performs stochastic gradient descent with momentum.
        % mu: Scalar between 0 and 1 giving the momentum value. Setting momentum = 0 reduces to sgd.
        % v: velocity var to store a moving average of the gradients.
        
            v = mu * v - obj.lr * obj.dw - obj.lr * obj.reg; % update moving gradients
            weightNew = obj.weightOld + v; % update weight            
        end
        
        function [weightNew,cache] = rmsprop(obj,cache)
        %  Uses the RMSProp update rule, which uses a moving average of squared gradient
        %  values to set adaptive per-parameter learning rates.
        
        %  decay_rate: Scalar between 0 and 1 giving the decay rate for the squared gradient cache.
        %  epsilon: Small scalar used for smoothing to avoid dividing by zero.
        %  cache: Moving average of second moments of gradients.
        decay_rate = 0.99;
        epsilon = 1e-7;
        cache = decay_rate * cache + (1 - decay_rate) * obj.dw ^ 2;
        weightNew = obj.weightOld - obj.lr * obj.dw /(sqrt(cache)+epsilon);
            
        end
        
        function [weightNew,m,v] = adam(obj,m,v,t)
        % Uses the Adam update rule, which incorporates moving averages of both the
        % gradient and its square and a bias correction term.    
        
        %  m: Moving average of gradient.
        %  v: Moving average of squared gradient.
        %  t: Epoch number
        
        beta1 = 0.9;
        beta2 = 0.999;
        epsilon = 1e-8;
        
        m = beta1 * m + (1 - beta1) * obj.dw;
        v = beta2 * v + (1 - beta2) * (obj.dw ^ 2);
        
        % bias correction:
        mb = m / (1 - beta1^t);
        vb = v / (1 - beta2^t);
        weightNew = obj.weightOld - obj.lr * mb /(sqrt(vb)+epsilon);
        end
    end
end