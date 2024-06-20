function nssvdd = nssvddtrain(Traindata,params)
%nssvddtrain() is a function for training a model based on "Newton Method-Based Subspace Support Vector Data Description"
% Input
%    Traindata = Contains training data from a single (target) class for training a model.
%    params
% params.C        :Value of hyperparameter C, Default=0.1.
% params.d        :Data in lower dimension, make sure that params.dim<D, Default=2.
% params.eta      :Needed only with gradient solution, Used as step size for gradient, Default=0.01.
% params.npt      :Used for selecting non-linear data description. Possible options are 1 (for non-linear data description), default=1 (linear data description)
% params.s        :Hyperparameter for the kernel, used in non-linear data description. Default=10.
% params.minmax   :Possible options are 'max', 'min' ,Default='min'.
% params.maxIter  :Maximim iteraions of the algorithm. Default=10.
% params.consType :Regularizatioin term.
% params.bta      :Controling the importance of regularization term (psi)
%
%
% Output      :nssvdd.modelparam = Trained model (for every iteration)
%             :nssvdd.Q= Projection matrix (after every iteration)
%             :nssvdd.npt=non-linear train data information, used for testing data

%% Setting Default parameters
default_params.C = 0.1;
default_params.d = 2;
default_params.eta = 0.01;
default_params.npt = 1;
default_params.s = 10;
default_params.minmax = 'max';
default_params.maxIter = 10;
default_params.consType=0;
default_params.bta =0;
%%
given = fieldnames(params);
defaults = fieldnames(default_params);
missingIdx = find(~ismember(defaults, given));
% Assign missing fields to params
for i = 1:length(missingIdx)
    params.(defaults{missingIdx(i)}) = default_params.(defaults{missingIdx(i)});
end


if params.d > size(Traindata,1)
    error('d must be <= D')
end

if params.C < 1/size(Traindata,2)
    error( 'C should be larger than 1/N');
end

if params.npt==1
    disp('Non-linear NSSSVDD running...')
    %RBF kernel
    N = size(Traindata,2);
    Dtrain = ((sum(Traindata'.^2,2)*ones(1,N))+(sum(Traindata'.^2,2)*ones(1,N))'-(2*(Traindata'*Traindata)));
    sigma2 = params.s * mean(mean(Dtrain));  A = 2.0 * sigma2;
    Ktrain_exp = exp(-Dtrain/A);
    % % center_kernel_matrices
    N = size(Ktrain_exp,2);
    Ktrain = (eye(N,N)-ones(N,N)/N) * Ktrain_exp * (eye(N,N)-ones(N,N)/N);
    [evcs,evls] = eig(Ktrain);        evls = diag(evls);
    [U, s] = sortEigVecs(evcs,evls);  S = diag(s);
    s_acc = cumsum(s)/sum(s);
    II = find(s_acc>=0.99);
    LL = II(1);
    Phi = ( S(1:LL,1:LL)^(0.5) * U(:,1:LL)' );
    %Saving useful variables for non-linear testing
    npt_data={1,A,Ktrain_exp,Phi,Traindata};%1,A,Ktrain,Phi,Traindata (1 is for flag)
    Traindata=Phi;
else
    disp('Linear NSSVDD running...')
end

[D,N] = size(Traindata);

St = eye(D);

Q = initialize_Q(size(Traindata,1),params.d);

for ii=1:params.maxIter
    
    SS = pinv(real(sqrtm(Q*St*Q' + 10^-6*eye(size(Q,1))))); % e inverse sqrroot
    reducedData=SS*Q*Traindata;
    Model = svmtrain(ones(N,1), reducedData', ['-s ',num2str(5),' -t 0 -c ',num2str(params.C)]);
    Qiter{ii}=SS*Q;
    Modeliter{ii}=Model;
    
    Alphavector=fetchalpha(Model,N);
    La = diag(Alphavector) - Alphavector*Alphavector';
    
    %Update Q
     Q=newton_update(Q,La,Traindata,Alphavector,params);
    
    if isempty(Q)
        break;
    end
    
    %orthogonalize and normalize Q
    Q = OandN_Q(Q);
end

nssvdd.modelparam = Modeliter;
nssvdd.Q = Qiter;

if params.npt==1
    nssvdd.npt=npt_data;
else
    nssvdd.npt{1}=0;
end
end
