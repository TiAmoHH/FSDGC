function [idxa] = Self_structure_X(X,num_self)
num_k = 5; %自监督的次数
[n,~] = size(X);
A = sparse(n,n);
for iter_k = 1 : num_k
    [idx, ~] = kmeans(X,num_self,'MaxIter',10,'emptyaction','singleton');
%     idx = kmeans(X,num_self, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
    H = sparse(n,num_self);
    for i = 1 : n
       H(i,idx(i)) = 1; 
    end
    A = A+H*H';
end
% idA = find(A>=num_k/2);
% A0 = zeros(n,n);
% A0(idA) = A(idA);
A(A <(num_k*0.6)) = 0;
[connectivitynum, id]=graphconncomp(A);
idxa = cell(connectivitynum,1);
for i = 1:connectivitynum
   a = find(id==i);
   idxa{i} = a;
end


end