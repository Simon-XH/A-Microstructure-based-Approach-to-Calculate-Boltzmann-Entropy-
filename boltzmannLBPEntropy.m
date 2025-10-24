function [BE_based_LBP,T_C] = boltzmannLBPEntropy(I,radius,neighbor_model,uniform_model)
%BOLTZMANNLBPENTROPY 基于遥感图像的微观结构计算玻尔兹曼熵
%   输入参数    I:遥感图像 RGB图像转换为灰度图像计算   默认 （I）=（I,1,1,0）
%         radius:LBP半径，默认为1
% neighbor_model:LBP邻域模式，默认正方形邻域1
%  uniform_model:是否考虑均匀模式进行去噪， 默认不考虑均匀LBP模式0

%  输出参数：   图像的玻尔兹曼熵，玻尔兹曼熵常数取1，log底数取2，因此单位化为bit

%% I = imread('dataset/cat100.jpg');

I = double(I);
[rowI,colI,depth] = size(I);

if ~exist('radius','var')
    radius = 1;    %默认
end

if ~exist('neighbor_model','var')
    neighbor_model = 1;    %默认正方形邻域模式
end

if ~exist('uniform_model','var')
    uniform_model = 0;    %默认不考虑均匀LBP模式
end

if depth > 2
    msgbox('Warning:输入影像波段大于1，转换为灰度影像进行计算！','警告');
    I = rgb2gray(I);
end
if radius <= 0 || (2*radius + 1) > min(rowI,colI)
    error('邻域半径： r为整数, 且 0 < 2r+1 < min(size(img))');
end

neighbors = 8;   % 只采用8个方向的邻域像素

%%%---在图像I边缘填充背景值0，使得LBP特征图的维度与原始输入影像I的一样-------------
Ileftright = zeros(rowI,radius);
newI = [Ileftright I Ileftright];
Itopbottom = zeros(radius,colI+2*radius);
newI = [Itopbottom; newI; Itopbottom];
I = newI;

%%%--------------------------计算玻尔兹曼熵--------------------------------------
[h,w] = size(I);
LBP_W = zeros(h-2*radius, w-2*radius);  %各个窗口下的微观态数目
% LBP_Value = LBP_W;                      %各个窗口下的LBP值（十进制）
LBP_U = LBP_W;                          %各个窗口下的LBP点跳变数目，U={0,2,4,6} ,U<=2 是均匀模式，否则 非均匀模式

tic;

for i = radius + 1:1:h - radius
    %     disp(['i=' num2str(i) '/' num2str(h-radius) ',--->>>Function:CalculteImgLBP<<<---']);
    for j = radius + 1:1:w-radius
        % 获得中心像素点的灰度值
        centerpixel = I(i,j);
        neighbor = zeros(1,neighbors);  %邻域灰度值向量
        for k = 1:neighbors
            if neighbor_model == 0
                %% 按圆形LBP计算第k个采样点的灰度值
                % 计算采样点对于中心点坐标的偏移量rx，ry
                rx = radius * cos(2.0 * pi * k / neighbors);
                ry = -(radius * sin(2.0 * pi * k / neighbors));
                % 为双线性插值做准备
                % 对采样点偏移量分别进行上下取整
                x1 = floor(rx);
                x2 = ceil(rx);
                y1 = floor(ry);
                y2 = ceil(ry);
                % 将坐标偏移量映射到0-1之间
                tx = rx - x1;
                ty = ry - y1;
                % 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
                w1 = (1-tx) * (1-ty);
                w2 =    tx  * (1-ty);
                w3 = (1-tx) *    ty;
                w4 =    tx  *    ty;
                % 根据双线性插值公式计算第k个采样点的灰度值
                neighbor(k) = I(i+y1,j+x1) * w1 + I(i+y2,j+x1) *w2 + I(i+y1,j+x2) *  w3 +I(i+y2,j+x2) *w4;
            else
                %% 按正方形LBP计算第k个采样点的灰度值
                rx = cos(2.0 * pi * k / neighbors);
                ry = -sin(2.0 * pi * k / neighbors);
                x_square = round(rx) * radius;
                y_square = round(ry) * radius;
                neighbor(k) = I(i+x_square,j+y_square);
            end
        end
        % LBP的微观态数目图像
        LBP_W(i-radius,j-radius) =  computeLBPW(neighbor,centerpixel);
        
        if uniform_model
            LBP_U(i-radius,j-radius) =  computeLBPU(neighbor,centerpixel);
        end
    end
end

%--计算玻尔兹曼熵-------------------------------------------------------------------
% Prod_W11 = prod(W);      % 图像微观态的总数目,直接累积会数值溢出！
% 由于 log(w(1)*w(2)*w(n))=log(w(1))+log(w(2))+log(w(n)); 因此采用累加计算klogW

W = LBP_W(:);
num_kernel = (h-2*radius)*(w-2*radius);
log2_Prod_W = zeros(1,num_kernel);
for i=1:num_kernel
    log2_Prod_W(i) = log2(W(i));      %以bit作为单位
end

k_b=1;                  % 玻尔兹曼熵常数，取1
%be = k_b*log2(Prod_W);
BE_based_LBP = k_b*sum(log2_Prod_W); % s = klnW

%%------均匀模式下的玻尔兹曼熵----------扣除非均匀模式窗口的微观态数目
if uniform_model
    U = LBP_U(:);
    log2_Prod_W_new = zeros(1,num_kernel);
    for i = 1:num_kernel
        if U(i) <= 2            
            log2_Prod_W_new(i) = log2(W(i));      %以bit作为单位
        end
    end
    BE_based_LBP = k_b*sum(log2_Prod_W_new); % s = klog2(W)    扣除非常见纹理结构处的玻尔兹曼熵
end


disp(['----End of LBP Boltzmann Entropy Process,Time=',num2str(toc),'----']);

T_C = toc;

end

%% --------子函数----------------------------------------------
function LBP_W_Value =  computeLBPW(neighbor,centerpixel)
%% 计算窗口微观态数目
len = length(neighbor);
minN = min([neighbor centerpixel]);
maxN = max([neighbor centerpixel]);

LBP_W_Neighbor = zeros(1,len);
for i = 1:len
    if neighbor(i) >= centerpixel
        LBP_W_Neighbor(i) = maxN - centerpixel + 1;
    else
        LBP_W_Neighbor(i) = centerpixel - minN;
    end
end
LBP_W_Value = cumprod(LBP_W_Neighbor,"omitnan");
LBP_W_Value = LBP_W_Value(len);
end


function LBP_U = computeLBPU(neighbor,centerpixel)
%% 计算跳变数目

len = length(neighbor);
LBP2 = zeros(1,len);

for i = 1:len
    LBP2(i) = coms(neighbor(i),centerpixel);
end
LBP2Next = LBP2;
LBP2Next = [LBP2Next LBP2Next(1)];
LBP2Next(1) = [];

LBP_U = sum(abs(LBP2Next - LBP2));

end

function flag = coms(a,b)
if a-b<0
    flag = 0;
else
    flag = 1;
end
end