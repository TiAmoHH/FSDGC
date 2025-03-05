function [F,G,obj,iter,converge,time] = FSDGC(B,F,G,c,idxa,ITER)
% tic
if nargin < 6
    ITER = 20;      % Maxiter 20
end

% [n,m] = size(B);
% num_self = size(idxa);
%% 
for k = 1:c
    P(:,k) = F(:,k)./sqrt(F(:,k)'*F(:,k)+eps);
    Q(:,k) = G(:,k)./sqrt(G(:,k)'*G(:,k)+eps);
%     fc = length(find(fl==k));   % O(1)
%     P(:,k) = F(:,k)./sqrt(fc);  % O(n_k) 
%     gc = length(find(gl==k));   % O(1)
%     Q(:,k) = G(:,k)./sqrt(gc);  % O(m_k) 
end 
obj = zeros(ITER+1,1);
obj(1) = trace(P'*B*Q);

%% First optimization to facilitate subsequent optimizations
% update G,fix F
V = B'*P;
[G,gg,obj_G,changed_g,converge_g] = updateGG(V,G);     % O(mct) t<10
for cc = 1:c                 
        Q(:,cc) = G(:,cc)./sqrt(gg(cc)+eps);   % O()
end
% update F,fix G
U = B*Q; 
[F,ff,obj_F,converge_f] = Init(U,F,idxa);

for cc = 1:c
        P(:,cc) = F(:,cc)./sqrt(ff(cc)+eps);   % O(1)
end

obj(2) = trace(P'*B*Q);
converge = true;
tic;
%% Optimization
for iter = 1:ITER
    % update G,fix F
     V = B'*P;                                  % O(mn)
    [G,gg,~,~,converge_g] = updateGG(V,G);     % O(mct) t<10
    for cc = 1:c                 
        Q(:,cc) = G(:,cc)./sqrt(gg(cc)+eps);   % O()
    end
%     tic
    % update F,fix G
    U = B*Q;                                   % O(nc)
    [F,ff,~,~,converge_f] = updateFF(U,F,idxa);     % O(nct) t<10
    for cc = 1:c
        P(:,cc) = F(:,cc)./sqrt(ff(cc)+eps);   % O(1)
    end
%     toc
    % Record Objs
    obj(iter+2) = trace(P'*B*Q);
    % Converge
    err_obj = obj(iter+2)-obj(iter+1);
    per_obj = err_obj/obj(iter+1);
    if err_obj < 0
        converge = false;
    end
%     if h>2 && abs(err_obj)<1e-5
    if iter>2 && per_obj<1e-5
        break;
    end
end

time = toc;






end






%% Function
function [F,ff,obj_F,changed,converge_f] = updateFF(U,F,idxa) % O(pc)
%% Preliminary
[~,c] = size(F);
num_self = size(idxa);
obj_F = zeros(11,1);           

ff = sum(F);                        % O(nc)
uf = sum(U.*F);                     % O(nc)

up = zeros(1,c);
for cc = 1:c                        % O(c)
    up(cc) = uf(cc)./sqrt(ff(cc));  % O(1)
end
obj_F(1) = sum(up);                 % objf

[~,label]=max(F, [], 2);
changed = zeros(10,1);
incre_F = zeros(1,c);
converge_f = true;
%% Update
for iterf = 1:10                    % O(nct) t<5
    converged = true;
    for i = 1:num_self
        idxi = idxa{i};
        s = length(idxi);
        ui = sum(U(idxi,:),1);
        label_idxi = label(idxi);
        only = numel(unique(label_idxi)) == 1;  % 检查唯一值的数量是否为 1
        if ~only
           continue; 
        end
        id0 = label_idxi(1);
        if ff(id0) == s
           continue; 
        end
        
        
        
        for k = 1:c                          % O(c)
            if k == id0
                incre_F(k) = uf(k)/sqrt(ff(k)+eps) - (uf(k) - ui(k))/sqrt(ff(k)-s+eps);
            else
                incre_F(k) = (uf(k)+ui(k))/sqrt(ff(k)+s+eps) - uf(k)/sqrt(ff(k)+eps);
            end
        end

        [~,id] = max(incre_F);
        if id~=id0                           % O(1)
            converged = false;               
            changed(iterf) = changed(iterf)+1;
            F(idxi,id0) = zeros(s,1); F(idxi,id) = ones(s,1);
            ff(id0) = ff(id0) - s;           % id0 from 1 to 0, number -1
            ff(id)  = ff(id) + s;            % id from 0 to 1，number +1
            uf(id0) = uf(id0) - ui(id0);
            uf(id)  = uf(id) + ui(id);
            label(idxi) = ones(s,1)*id;
        end
    end
    if converged
        break;
    end

    for cc = 1:c
        up(cc) = uf(cc)/sqrt(ff(cc)+eps);
    end
    obj_F(iterf+1) = sum(up);

    err_obj_f = obj_F(iterf+1)-obj_F(iterf);
    if err_obj_f < 0
        converge_f = false;
    end
end
end



function [G,gg,obj_G,changed,converge] = updateGG(V,G) % O(mc)

[m,c] = size(G);
obj_G = zeros(11,1);

gg = sum(G)+eps*ones(1,c);     % tr(GTG) O(mc)
gv = sum(V.*G);                % tr(VTG) O(mc)

qv = zeros(1,c);
for cc = 1:c                        % O(c)
    qv(cc) = gv(cc)./sqrt(gg(cc));  % O(1)
end
obj_G(1) = sum(qv);            % objg O(c)

changed = zeros(10,1);
incre_G = zeros(1,c);
converge = true;
%% Update
for iterg = 1:10               % O(mct) t<10
    converged = true;
    for i = 1:m                           % O(mc)
        vi = V(i,:);
        [~,id0] = find(G(i,:)==1);
        if gg(id0) == 1
           continue; 
        end
        
        for k = 1:c                       % O(c)
            if k == id0
                incre_G(k) = gv(k)/sqrt(gg(k)+eps) - (gv(k) - vi(k))/sqrt(gg(k)-1+eps);
            else
                incre_G(k) = (gv(k)+vi(k))/sqrt(gg(k)+1+eps) - gv(k)/sqrt(gg(k)+eps);
            end
        end

        [~,id] = max(incre_G);
        %         [~,id] = max(incre_g);     % 该行对应样本更新后的归属类别 1*1
        if id~=id0
            converged = false;               % not converge
            changed(iterg) = changed(iterg)+1; % change record
            G(i,id0) = 0;G(i,id) = 1;
            gg(id0) = gg(id0) - 1;           % id0 from 1 to 0, number -1
            gg(id)  = gg(id) + 1;            % id from 0 to 1, number +1
            gv(id0) = gv(id0) - vi(id0);     % id0 from 1 to 0, update gv
            gv(id)  = gv(id) + vi(id);       % id from 0 to 1, update gv
        end
    end
    if converged                             % m anchors traversal, false continue, true break
        break;
    end

    %% Obj tr(VT*Q)
    for cc = 1:c
        qv(cc) = gv(cc)/sqrt(gg(cc)+eps);
    end
    obj_G(iterg+1) = sum(qv);

    err_obj_g = obj_G(iterg+1)-obj_G(iterg);
    if err_obj_g < 0
        converge = false;
    end
end

end





function [F,ff,obj_F,converge_f] = Init(U,F,idxa) 
[~,c] = size(F);
num_self = numel(idxa);
obj_F = zeros(11,1); 
ff = sum(F);                        % O(nc)
uf = sum(U.*F);                     % O(nc)
up = zeros(1,c);
for cc = 1:c                        % O(c)
    up(cc) = uf(cc)./sqrt(ff(cc));  % O(1)
end
obj_F(1) = sum(up);    
delta_F = zeros(1,c);
converge_f = true;
for iterf = 1:10
    converged = true;
    for i = 1:num_self
       idxi = idxa{i};
       s = length(idxi);
       for k = 1:c
           f_k = F(:,k);
           id0 = find(f_k(idxi)==1);
           id1 = find(f_k(idxi)==0);
           
           uk1 = sum(U(idxi(id1),k));
           uk0 = sum(U(idxi(id0),k));
           delta_F(k) = (uf(k)+uk1)/sqrt(ff(k)+(s-sum(f_k(idxi)))+eps) - (uf(k)-uk0)/sqrt(ff(k)-sum(f_k(idxi))+eps);

       end
       [~,q] = max(delta_F);
       for j = 1 : s
          [~,p] = find(F(idxi(j),:)==1);
          if q~=p
             converged = false; 
             F(idxi(j),p) = 0;
             F(idxi(j),q) = 1;
            ff(p) = ff(p) - 1;           % id0 from 1 to 0, number -1
            ff(q) = ff(q) + 1; 
            uf(p) = uf(p) - U(idxi(j),p);
            uf(q)  = uf(q) + U(idxi(j),q);
          end
       end
    end
    if converged
        break;
    end
    for cc = 1:c
       up(cc) = uf(cc)/sqrt(ff(cc)+eps);
    end
    obj_F(iterf+1) = sum(up);
    err_obj_f = obj_F(iterf+1)-obj_F(iterf);
    if err_obj_f < 0
        converge_f = false;
    end
end

end
