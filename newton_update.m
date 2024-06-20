function Q = newton_update(Q,La,Traindata,Alphavector,params)

if strcmp(params.minmax,'min'), eta = params.eta; else, eta = - params.eta; end

if params.consType==0
    const=0;
    const_hess=0;
elseif params.consType==1
    const_hess=2*params.bta*(Traindata*Traindata');
    const= Q*const_hess;
elseif params.consType==2
    const_hess=2*params.bta*(Traindata*(Alphavector*Alphavector')*Traindata');
    const= Q*const_hess;
elseif params.consType==3
    Alphavector_C=Alphavector;
    Alphavector_C(Alphavector_C==params.Cval)=0;
    const_hess=2*params.bta*Traindata*(Alphavector_C*Alphavector_C')*Traindata';
    const= Q*const_hess;
else
msg='Only psi 0,1,2 or 3 is possible';
error(msg)
end

Sa = Traindata*La*Traindata';

    Grad = 2*Q*Sa+(params.bta*const);
    Inv_GradGrad=pinv(2*Sa+const_hess);
    Q = Q' - eta*(Inv_GradGrad*Grad');
    Q=Q';

end


