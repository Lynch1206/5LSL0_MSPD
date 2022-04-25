function [ filter ] = update_filter( filter,e )

x=filter.x_delayed;
w_old = filter.w;
r=filter.r;
filter_type=filter.type;
sigma_x=filter.sigma_x;
filter.w=w_old; %default output

%% A1 scenario 1:f
if strcmpi(filter_type,'SGD')
    %implement the SGD update rule here
    alpha=filter.adaptation_constant;
    Rx = [2 -1; -1 2]; % see 1.2.1.a
    rex = [0;3]; % see 1.2.1.a
    filter.w = filter.w + 2 * alpha * (rex - Rx*w_old);
end

%% A1 scenario 1:i
if strcmpi(filter_type,'Newton')
    %implement the Newton update rule here
    alpha=filter.adaptation_constant;
    Rx = [2 -1; -1 2]; % see 1.2.1.a
    rex = [0;3]; % see 1.2.1.a
    filter.w=filter.w + 2 * alpha * (Rx^-1) * (rex - Rx*w_old);
end

%% A1 scenario 2:a
if strcmpi(filter_type,'LMS')
    %implement the LMS update rule here
    alpha=filter.adaptation_constant;
    filter.w=filter.w+2*alpha*x*r;
end

%% A1 scenario 2:b
if strcmpi(filter_type,'NLMS')
    %implement the NLMS update rule here
    alpha=filter.adaptation_constant;
    filter.w=filter.w+2*alpha/sigma_x*x*r;
end

%% A1 scenario 2:d
if strcmpi(filter_type,'RLS')
    %implement the RLS update rule here
    lambda=filter.adaptation_constant;   
    gamma=1-1e-4;
    Rx_inv_old = filter.Rx_inv;
    g=(Rx_inv_old * filter.x)/(gamma^2 + filter.x'*Rx_inv_old*filter.x);
    filter.Rx_inv = gamma^-2 * (Rx_inv_old - g*filter.x'*Rx_inv_old);
    filter.rex = gamma^2 * filter.rex + filter.x * filter.e
    filter.w = filter.Rx_inv * filter.rex;
end

%% A1 scenario 2:e
if strcmpi(filter_type,'FDAF')
    %implement the FDAF update rule here
    alpha=filter.adaptation_constant;
    beta = 0.5;
    X = filter.F * filter.x_delayed;
    Xi = X.*eye(filter.length);
    filter.P = beta * filter.P + (1-beta) * abs(Xi)^2 / filter.length;
    filter.W = filter.W+ 2 * alpha*filter.P^-1 * conj(X) * r;
    filter.w = filter.F*filter.W;
end


end

