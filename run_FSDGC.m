function [label,F,G,obj,iter,converge,time] = run_FSDGC(X,B,c,num_self,Y,F,G)



%% Construct self-supervised information
idxa = Self_structure_X(X,num_self);
% idxa = Self_structure_gt(Y,c);
%% 
[F,G,obj,iter,converge,time] = FSDGC(B,F,G,c,idxa);
[~,label] = max(F, [], 2);



end