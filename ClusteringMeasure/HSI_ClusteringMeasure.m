function [PA, UA, AA, OA, Kappa] = HSI_ClusteringMeasure(Y,predY)
%   按照最近邻分类器对测试样本进行分类，并计算
%   混淆矩阵 cf_matrix, 平均分类精度 Average classification Accuracy (AA)
%   总体分类精度 Overall classification Accuracy (OA), Kappa coefficient (kappa)

predY = bestMap(Y,predY);

cf_matrix = confusionmat(Y,predY); % 混淆矩阵

N = sum(sum(cf_matrix));    % 混淆矩阵的总数
col_num = sum(cf_matrix,1); % 混淆矩阵的列总数（行向量）
row_num = sum(cf_matrix,2); % 混淆矩阵的行总数（列向量）

PA = diag(cf_matrix)./col_num';  % 生产者精度 

UA = diag(cf_matrix)./row_num;  % 使用者精度 

AA = sum(UA)/size(UA,1);

OA = trace(cf_matrix)/N;  % 总分类精度

Kappa = (N*trace(cf_matrix)-col_num*row_num)/(N^2-col_num*row_num);

end



