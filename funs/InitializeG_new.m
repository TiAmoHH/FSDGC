function [Ini,inG0] = InitializeG_new(n,c,iter)
% Initialize clustering indicator matrix G
% Ini denotes the initialize label vector
rng(iter,'twister');
I = eye(c);
R = randperm(size(I,1));
inG0 = I(R,:);
r = n - floor(n/c)*c;
j = floor(n/c);
if r == 0
    for i=1:j-1
        R = randperm(size(I,1));
        A = I(R,:);
        inG0=[inG0;A];
    end
else
    for i=1:j-1
        R = randperm(size(I,1));
        A = I(R,:);
        inG0=[inG0;A];
    end
    for i=1:r
        R = randperm(size(I,1));
        a=I(R(1),:);
        inG0=[inG0;a];
    end
end
[row,col] = find(inG0);  % 非零元素的行和列索引
[~,idx] = sort(row,'ascend');
Ini = col(idx);

end