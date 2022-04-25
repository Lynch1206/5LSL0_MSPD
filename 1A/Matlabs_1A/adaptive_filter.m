classdef adaptive_filter
    %Performs sample-wise adaptive filtering
    %   input x and e, outputs r
    
    properties
        length;  %number of filter coefficients
        type;    %filter type
        adaptation_constant;
        w=[];           %filter coefficients
        x_delayed = []; %delayed input signals
        w_history = []; %keep track of filter coefficients as they evolve
        r;
        
        %add your own variables here
        sigma_x;
        F;
        F_inv;
        P;
        W;
        
    end
    
    methods
        function obj = adaptive_filter(length,type,adaptation_constant)
            if(nargin>0)
            obj.length=length;
            obj.type=type;
            obj.adaptation_constant=adaptation_constant;
            obj.w=zeros(length,1);
            obj.x_delayed=zeros(length,1);
            obj.w_history=zeros(0,length);
            
            %initialize your variables here
            obj.sigma_x = 0;
            obj.F=dftmtx(length);
            obj.F_inv = obj.F^-1;
            obj.P = eye(length);
            obj.W = obj.F_inv * obj.w;

            end
        end
        function obj = filter(obj,x,e)
            obj.x_delayed=[x ; obj.x_delayed(1:obj.length-1)];
            obj.sigma_x = mean(obj.x_delayed.^2);
            e_hat=obj.w.' * obj.x_delayed;
            obj.r=e-e_hat;
            obj.w_history=[obj.w_history ; obj.w.']; %you may want to remove this line to gain speed
            obj=update_filter(obj,e);        
        end
    end
    
end

