function [BE_based_LBP,T_C] = boltzmannLBPEntropy(I,radius,neighbor_model,uniform_model)
%BOLTZMANNLBPENTROPY ����ң��ͼ���΢�۽ṹ���㲣��������
%   �������    I:ң��ͼ�� RGBͼ��ת��Ϊ�Ҷ�ͼ�����   Ĭ�� ��I��=��I,1,1,0��
%         radius:LBP�뾶��Ĭ��Ϊ1
% neighbor_model:LBP����ģʽ��Ĭ������������1
%  uniform_model:�Ƿ��Ǿ���ģʽ����ȥ�룬 Ĭ�ϲ����Ǿ���LBPģʽ0

%  ���������   ͼ��Ĳ��������أ����������س���ȡ1��log����ȡ2����˵�λ��Ϊbit

%% I = imread('dataset/cat100.jpg');

I = double(I);
[rowI,colI,depth] = size(I);

if ~exist('radius','var')
    radius = 1;    %Ĭ��
end

if ~exist('neighbor_model','var')
    neighbor_model = 1;    %Ĭ������������ģʽ
end

if ~exist('uniform_model','var')
    uniform_model = 0;    %Ĭ�ϲ����Ǿ���LBPģʽ
end

if depth > 2
    msgbox('Warning:����Ӱ�񲨶δ���1��ת��Ϊ�Ҷ�Ӱ����м��㣡','����');
    I = rgb2gray(I);
end
if radius <= 0 || (2*radius + 1) > min(rowI,colI)
    error('����뾶�� rΪ����, �� 0 < 2r+1 < min(size(img))');
end

neighbors = 8;   % ֻ����8���������������

%%%---��ͼ��I��Ե��䱳��ֵ0��ʹ��LBP����ͼ��ά����ԭʼ����Ӱ��I��һ��-------------
Ileftright = zeros(rowI,radius);
newI = [Ileftright I Ileftright];
Itopbottom = zeros(radius,colI+2*radius);
newI = [Itopbottom; newI; Itopbottom];
I = newI;

%%%--------------------------���㲣��������--------------------------------------
[h,w] = size(I);
LBP_W = zeros(h-2*radius, w-2*radius);  %���������µ�΢��̬��Ŀ
% LBP_Value = LBP_W;                      %���������µ�LBPֵ��ʮ���ƣ�
LBP_U = LBP_W;                          %���������µ�LBP��������Ŀ��U={0,2,4,6} ,U<=2 �Ǿ���ģʽ������ �Ǿ���ģʽ

tic;

for i = radius + 1:1:h - radius
    %     disp(['i=' num2str(i) '/' num2str(h-radius) ',--->>>Function:CalculteImgLBP<<<---']);
    for j = radius + 1:1:w-radius
        % ����������ص�ĻҶ�ֵ
        centerpixel = I(i,j);
        neighbor = zeros(1,neighbors);  %����Ҷ�ֵ����
        for k = 1:neighbors
            if neighbor_model == 0
                %% ��Բ��LBP�����k��������ĻҶ�ֵ
                % ���������������ĵ������ƫ����rx��ry
                rx = radius * cos(2.0 * pi * k / neighbors);
                ry = -(radius * sin(2.0 * pi * k / neighbors));
                % Ϊ˫���Բ�ֵ��׼��
                % �Բ�����ƫ�����ֱ��������ȡ��
                x1 = floor(rx);
                x2 = ceil(rx);
                y1 = floor(ry);
                y2 = ceil(ry);
                % ������ƫ����ӳ�䵽0-1֮��
                tx = rx - x1;
                ty = ry - y1;
                % ����0-1֮���x��y��Ȩ�ؼ��㹫ʽ����Ȩ�أ�Ȩ�����������λ���޹أ��������Ĳ�ֵ�й�
                w1 = (1-tx) * (1-ty);
                w2 =    tx  * (1-ty);
                w3 = (1-tx) *    ty;
                w4 =    tx  *    ty;
                % ����˫���Բ�ֵ��ʽ�����k��������ĻҶ�ֵ
                neighbor(k) = I(i+y1,j+x1) * w1 + I(i+y2,j+x1) *w2 + I(i+y1,j+x2) *  w3 +I(i+y2,j+x2) *w4;
            else
                %% ��������LBP�����k��������ĻҶ�ֵ
                rx = cos(2.0 * pi * k / neighbors);
                ry = -sin(2.0 * pi * k / neighbors);
                x_square = round(rx) * radius;
                y_square = round(ry) * radius;
                neighbor(k) = I(i+x_square,j+y_square);
            end
        end
        % LBP��΢��̬��Ŀͼ��
        LBP_W(i-radius,j-radius) =  computeLBPW(neighbor,centerpixel);
        
        if uniform_model
            LBP_U(i-radius,j-radius) =  computeLBPU(neighbor,centerpixel);
        end
    end
end

%--���㲣��������-------------------------------------------------------------------
% Prod_W11 = prod(W);      % ͼ��΢��̬������Ŀ,ֱ���ۻ�����ֵ�����
% ���� log(w(1)*w(2)*w(n))=log(w(1))+log(w(2))+log(w(n)); ��˲����ۼӼ���klogW

W = LBP_W(:);
num_kernel = (h-2*radius)*(w-2*radius);
log2_Prod_W = zeros(1,num_kernel);
for i=1:num_kernel
    log2_Prod_W(i) = log2(W(i));      %��bit��Ϊ��λ
end

k_b=1;                  % ���������س�����ȡ1
%be = k_b*log2(Prod_W);
BE_based_LBP = k_b*sum(log2_Prod_W); % s = klnW

%%------����ģʽ�µĲ���������----------�۳��Ǿ���ģʽ���ڵ�΢��̬��Ŀ
if uniform_model
    U = LBP_U(:);
    log2_Prod_W_new = zeros(1,num_kernel);
    for i = 1:num_kernel
        if U(i) <= 2            
            log2_Prod_W_new(i) = log2(W(i));      %��bit��Ϊ��λ
        end
    end
    BE_based_LBP = k_b*sum(log2_Prod_W_new); % s = klog2(W)    �۳��ǳ�������ṹ���Ĳ���������
end


disp(['----End of LBP Boltzmann Entropy Process,Time=',num2str(toc),'----']);

T_C = toc;

end

%% --------�Ӻ���----------------------------------------------
function LBP_W_Value =  computeLBPW(neighbor,centerpixel)
%% ���㴰��΢��̬��Ŀ
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
%% ����������Ŀ

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