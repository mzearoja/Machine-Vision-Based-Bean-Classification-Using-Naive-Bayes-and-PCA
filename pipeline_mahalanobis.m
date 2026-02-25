folder = '\\nas01.itap.purdue.edu\puhome\My Documents\Lab3_beanImages';
filePattern = fullfile(folder, '*.tif');
srcFiles = dir(filePattern);
numImages = size(srcFiles,1);
 for k = 1:numImages;
    fprintf('%s\n',srcFiles(k).name);
    im = imread(srcFiles(k).name);
    R = im(:,:,1);
    G = im(:,:,2);
    B = im(:,:,3);
    [m,n,K] = size(im);
    im1 = double(reshape(im,m*n,K));
    im2 = normr(im1); % Normalization to remove the difference of intensity
    I = reshape(im2,m,n,K);
   r = I(:,:,1);
   g = I(:,:,2);
   b = I(:,:,3);
   gray = r+g-b; % Here, the color information was used to segment beans.
	% If still wanting to try Mahalanobis distance method, there's function pdist2(X,Y,'mahalanobis').
   
%pairwise comparison Yi. (Yi vs. all Xi values)') ylabel('Mahalanobis distance') legend('mahal()','pdist2()')

   level = graythresh(gray);
   bw = im2bw(gray,level);
   bw = bwmorph(bw,'majority');
   maskedRgbImage = bsxfun(@times, im, cast(bw, 'like',im));
   figure, imshow(maskedRgbImage);
     [L,num] = bwlabel(bw,4);
     number = 0.5*((r - g) + (r - b));  
    den = sqrt((r - g).^2 + (r - b).*(g - b));  
    theta = acos(number./(den + eps));
    H = theta;  
    H(b > g) = 2*pi - H(b > g);  
    H = H/(2*pi);  
    number = min(min(r, g), b);  
    den = r + g + b;  
    den(den == 0) = eps;  
    S = 1 - 3.* number./den;  
    H(S == 0) = 0;  
    I = (r + g + b)/3;  
    hsi = cat(3,H, S, I); 
    hsi = reshape(hsi,m*n,K);
        s = regionprops(bw,'Centroid','Area',...
        'Eccentricity','MajorAxisLength','MinorAxisLength',...
            'Perimeter','Solidity'); 
     switch k;
        case {1,2}
            T.class = 'AdukiBeans';
        case {3,4}
            T.class = 'BrownBeans';
        case {5,6}
            T.class = 'ChickPeas';
        case {7,8}
            T.class = 'GreenPeas';
         case {9,10}
            T.class = 'Lentils';  
         case {11,12}
            T.class = 'MarrowfatPeas';
         case {13,14}
            T.class = 'WhiteBeans';
         case {15,16}
            T.class = 'kidneybeans';
     end
    STATS=T;
    for j=1:num-1;
        STATS=cat(1,STATS,T);
    end
        for i=1:num;
        STATS(i).num=i;
        STATS(i).Area=s(i).Area;
        index=find(L==i);
        STATS(i).Hue = mean(hsi(index,1));
        STATS(i).Indensity=mean(hsi(index,3));
        STATS(i).Perimeter=s(i).Perimeter;
        STATS(i).MajorAxisLength=s(i).MajorAxisLength;
        STATS(i).MinorAxisLength=s(i).MinorAxisLength;
        STATS(i).Eccentricity=s(i).Eccentricity;
        STATS(i).Solidity=s(i).Solidity;
        thisBean=zeros(m*n,1);
        thisBean(index)=1;
        thisBean=reshape(thisBean,m,n);
        [L1,num1]=bwlabel(thisBean,8);
        d=0;
        while(num1>0)
            thisBean=imerode(thisBean,strel('diamond',1));
            d=d+1;
            [L1,num1]=bwlabel(thisBean,8);
        end
        STATS(i).Elongation=(STATS(i).Area)/((2*d)^2);
        STATS(i).AspectRatio=(STATS(i).MajorAxisLength)/(STATS(i).MinorAxisLength);
        STATS(i).Compactnes=(STATS(i).Perimeter)^2/(STATS(i).Area);
        STATS(i).Roundness=(STATS(i).Area)/pi/(STATS(i).Perimeter/2/pi+0.5)^2;
    end
    
   
    if(k == 1)  
        TSTATS=(struct2cell(STATS))';
    else
        TSTATS=cat(1,TSTATS,(struct2cell(STATS))');
    end
    
 end
    
 %%
 celltest = fieldnames(STATS)';
 cellout = [celltest;TSTATS];
 xlswrite('STATS.xlsx',cellout); % This 'cellout' table is our data table. 
 %%
 % This section is to calculate P-values
 % There're 8 different categories of beans. First kind of beans with remaing 7 kinds of beans.
 % Second kind of beans with remaining 6 kinds of beans......
 % Thus, a 'for' loop works well to achieve this.
%%
 % This section is to conduct Bayes classification.
 % For each kind of beans, 70% for training, 30% for testing.
Y(1:35,1) = {'AdukiBeans'};
Y(36:70,1) = {'BrownBean'};
Y(71:105,1) = {'ChickPea'};
Y(106:140,1) = {'GreenBean'};
Y(141:175,1) = {'Lentils'};
Y(176:210,1) = {'MarrowfatPea'};
Y(211:245,1) = {'WhiteBean'};
Y(246:280,1) = {'KidneyBean'};
X=zeros(400,12);

Yt(1:15,1) = {'AdukiBeans'};
Yt(16:30,1) = {'BrownBean'};
Yt(31:45,1) = {'ChickPea'};
Yt(46:60,1) = {'GreenBean'};
Yt(61:75,1) = {'Lentils'};
Yt(76:90,1) = {'MarrowfatPea'};
Yt(91:105,1) = {'WhiteBean'};
Yt(106:120,1) = {'KidneyBean'};
X=zeros(400,12);

for i=1:400
    for j=3:14
         X(i,j-2)= TSTATS{i,j};
    end
end
trainX = cat(1,X(1:35,:),...
X(51:85,:),...
X(101:135,:),...
X(151:185,:),...
X(201:235,:),...
X(251:285,:),...
X(301:335,:),X(351:385,:));
Mdl = fitcnb(trainX,Y);
predictX = X([36:50,86:100,136:150,186:200,236:250,286:300,336:350,386:400],:);
label = predict(Mdl,predictX);
% After obtaining the 'label', you need to write the Bayes classification results down. 
adukiBeansIndex = strcmp(Mdl.ClassNames,'AdukiBeans');
estimates = Mdl.DistributionParameters{adukiBeansIndex,1}
tabulate(Y)
hold on
Params = cell2mat(Mdl.DistributionParameters); 
Mu = Params(2*(1:3)-1,1:2); % Extract the means
Sigma = zeros(2,2,3);
figure
gscatter(trainX(:,1),trainX(:,2),Y);
h = gca;
cxlim = h.XLim;
cylim = h.YLim;
for j = 1:3
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    xlim = Mu(j,1) + 4*[-1 1]*sqrt(Sigma(1,1,j));
    ylim = Mu(j,2) + 4*[-1 1]*sqrt(Sigma(2,2,j));
    f = @(x1,x2)reshape(mvnpdf([x1(:),x2(:)],Mu(j,:),Sigma(:,:,j)),size(x1));
    fcontour(f,[xlim ylim]) % Draw contours for the multivariate normal distributions 
end
h.XLim = cxlim;
h.YLim = cylim;
title('Naive Bayes Classifier')
xlabel('train beans)')
ylabel('test')
legend('AdukiBean','BrownBean','ChickPea','Lentils','MarrowfatPea','WhiteBean','KidneyBean')
hold off


xlswrite('bayesResult.xlsx',bayesResult); % Write the Bayes results to excel
%% 
% This section is to conduct 'PCA'
[wcoeff,score,latent,~,explained] = pca(X);
coefforth = inv(diag(std(X)))* wcoeff;
coefforth*coefforth';
Xcentered = score*coeff';
biplot(coeff(:,1:2),'scores',score(:,1:3),'varlabels',{'v_1','v_2','v_3'v_1','v_2','v_3',});
score = score(:,1:3); % Here, select first 3 scores because the first 3 latent variables
   % explain more than 90% of the data 
trainScore = cat(1,score(1:35,:),...
score(51:85,:),...
score(101:135,:),...
score(151:185,:),...
score(201:235,:),...
score(251:285,:),...
score(301:335,:),score(351:385,:));
newMd = fitcnb(trainScore,Y);
predictScore = score([36:50,86:100,136:150,186:200,236:250,286:300,336:350,386:400],:);
newlabel = predict(newMd,predictScore);
scorebsPCA=Yt==string(newlabel);
xlswrite('bayesResultPCA.xlsx',scorebsPCA);
 % The final step is to write down the Bayes results after 'PCA'
 % Compare the results before and after 'PCA'. Try to explain the difference in your report.
%%
