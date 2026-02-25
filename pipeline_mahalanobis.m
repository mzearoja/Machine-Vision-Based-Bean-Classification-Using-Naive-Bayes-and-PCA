files = {
    'AdukiBeans1.tif';
    'AdukiBeans2.tif';
    'BrownBean1.tif';
    'BrownBean2.tif';
    'ChickPeas1.tif';
    'ChickPeas2.tif';    
    'GreenPea1.tif';
    'GreenPea2.tif';
    'kidneybean1.tif';
    'kidneybean2.tif';
    'Lentils1.tif';
    'Lentils2.tif';
    'MarrowfatPea1.tif';
    'MarrowfatPea2.tif';
    'WhiteBean1.tif';
    'WhiteBean2.tif';
    };
N=size(files,1);
for imagenum = 1:N
    fileStr = char(files{imagenum,1});
    Im = imread(fileStr, 'tif');

    % Segment image: here is just an example. You're encouraged to explore different seg methods as long as you think you find a good seg solution for your own project
    bean=roipoly(Im); % choose some background region
    [M,N,K]=size(Im);
    R=double(reshape(Im,M*N,K)); % if you use this function, please understand what it does here 
    idx=find(bean);
    R1=R(idx,1:K); % R1 now stores the list of RGB values of the pixels you just selected
    [C,m]=covmatrix(R1);
    % Color segmentation based on Mahalanobis distance.
    Seg=colorseg('mahalanobis',Im,150,m,C); % you need to put the p files in the same directory in order to call this function
    figure, imshow(Seg); % take a look at your result. If you're not satisfied with it please work on improving it with any methods you want to try.
    

    % Here is an example of how you can improve your seg result by applying morphologic operations
    Seg=bwmorph(Seg,'erode',3); % make sure you read through the help file for bwmorph function. You will find this function very useful in your future projects.
    Seg=bwmorph(Seg,'dilate',3);
    figure, imshow(Seg);
    % Label bean objects
    [L,num]=bwlabel(Seg,8); % now what is in the output L and num?
    num % run this line and see how many objects were recognized by your program?
    
    % Feature extraction
    hsi=rgb2hsi(Im);
    hsi=reshape(hsi,M*N,K);
    SS=regionprops(L,'Area','Perimeter','MajorAxisLength','MinorAxisLength','Eccentricity','Solidity');
    T.num=1; % now you start to store all your result into the structure T (please make sure you understand what a structure is in MATLAB).
    switch imagenum
    case {1,2}
        T.class='AdukiBeans'; % the bean class name here depends on the sequence you read them in. Please check and don't just simply copy the code here!
    case {3,4}
        T.class='BrownBean';
    case {5,6}
        T.class='GreenBean';
   case {7,8}
        T.class='ChickPea';
   case {9,10}
        T.class='KidneyBean';
   case {11,12}
        T.class='Lentils';
   case {13,14}
        T.class='MarrowfatPea';
   case {15,16}
        T.class='WhiteBean';
    end
    
    STATS=T;
    for i=1:num-1
        STATS=cat(1,STATS,T);
    end
    for beanNum = 1:num 
        % Extract features following definitions 
        STATS(beanNum).num=beanNum;
        STATS(beanNum).Area=SS(beanNum).Area;
        index=find(L==beanNum);
        STATS(beanNum).hue=mean(hsi(index,1));
        STATS(beanNum).indensity=mean(hsi(index,3));
        STATS(beanNum).Perimeter=SS(beanNum).Perimeter;
        STATS(beanNum).MajorAxisLength=SS(beanNum).MajorAxisLength;
        STATS(beanNum).MinorAxisLength=SS(beanNum).MinorAxisLength;
        STATS(beanNum).Eccentricity=SS(beanNum).Eccentricity;
        STATS(beanNum).Solidity=SS(beanNum).Solidity;
        thisBean=zeros(M*N,1);
        thisBean(index)=1;
        thisBean=reshape(thisBean,M,N);
        [L1,num1]=bwlabel(thisBean,8);
        d=0;
        while(num1>0)
            thisBean=imerode(thisBean,strel('diamond',1));
            d=d+1;
            [L1,num1]=bwlabel(thisBean,8);
        end
        STATS(beanNum).Elongation=(STATS(beanNum).Area)/((2*d)^2);
        STATS(beanNum).AspectRatio=(STATS(beanNum).MajorAxisLength)/(STATS(beanNum).MinorAxisLength);
        STATS(beanNum).Compactnes=(STATS(beanNum).Perimeter)^2/(STATS(beanNum).Area);
        STATS(beanNum).Roundness=(STATS(beanNum).Area)/pi/(STATS(beanNum).Perimeter/2/pi+0.5)^2;
        % Label bean objects
        [L,num]=bwlabel(Seg,8);
        beanNum
        % store features into a bean structure with bean label
 
    end    
    if(imagenum==1)
        TSTATS=(struct2cell(STATS))';
    else
        TSTATS=cat(1,TSTATS,(struct2cell(STATS))');
    end
    
end

xlswrite('STATS',TSTATS); % this is your feature file

%step 3
%check function ttest2() and use it to calculate the p values

%step 4
Y=zeros(400,12);

for(i=1:400)
    for(j=3:14)
        temp=TSTATS(i,j);
        Y(i,j-2)=temp{1,1};
    end
end
[C1,M1]=covmatrix(Y(1:35,:));
[C2,M2]=covmatrix(Y(51:85,:));
[C3,M3]=covmatrix(Y(101:135,:));
[C4,M4]=covmatrix(Y(151:185,:));
[C5,M5]=covmatrix(Y(201:235,:));
[C6,M6]=covmatrix(Y(251:285,:));
[C7,M7]=covmatrix(Y(301:335,:));
[C8,M8]=covmatrix(Y(351:385,:));

CA=cat(3,C1,C2,C3,C4,C5,C6,C7,C8);
MA=cat(2,M1,M2,M3,M4,M5,M6,M7,M8);
V=Y([36:50,86:100,136:150,186:200,236:250,286:300,336:350,386:400],:);
bayesclass=bayesgauss(V,CA,MA);
%get the classification table
Bayes=zeros(8,8);
for(i=1:120)
    Bayes(bayesclass(i,1),ceil(i/15))=Bayes(bayesclass(i,1),ceil(i/15))+1;
end
xlswrite('Bayes',Bayes);

% Principal component analysis
P=princomp(Y,6);% The features were selected to six here.

[C1,M1]=covmatrix(P.Y(1:35,:));
[C2,M2]=covmatrix(P.Y(51:85,:));
[C3,M3]=covmatrix(P.Y(101:135,:));
[C4,M4]=covmatrix(P.Y(151:185,:));
[C5,M5]=covmatrix(P.Y(201:235,:));
[C6,M6]=covmatrix(P.Y(251:285,:));
[C7,M7]=covmatrix(P.Y(301:335,:));
[C8,M8]=covmatrix(P.Y(351:385,:));

CA=cat(3,C1,C2,C3,C4,C5,C6,C7,C8);
MA=cat(2,M1,M2,M3,M4,M5,M6,M7,M8);
V=P.Y([36:50,86:100,136:150,186:200,236:250,286:300,336:350,386:400],:);
PCAclass=bayesgauss(V,CA,MA);
%get the classification table
PCA=zeros(8,8);
for(i=1:120)
    PCA(PCAclass(i,1),ceil(i/15))=PCA(PCAclass(i,1),ceil(i/15))+1;
end
xlswrite('PCA',PCA);

