path = 'Final.txt';     %path of training data
A = textread((path)); 
col = 1;                %defines the bandwith to calculate 1: color, 2: texture, 3: arrangement

%-----------
countMerged = 0;
countNotMerged =0;
for i=1:size(A,1)
    if A(i,4) == 1
        countMerged = countMerged+1;
    else
        countNotMerged = countNotMerged+1;
    end
end

PriorMerged = countMerged/size(A,1);
PriorNotMerged = countNotMerged/size(A,1);

Bmerged = zeros(1,1);
rBmerged = 1;
BNotmerged = zeros(1,1);
rBNotmerged = 1;
for i=1:size(A,1)   %go over rows
    if A(i,4)==1
        Bmerged(rBmerged,1) = A(i,1);
        Bmerged(rBmerged,2)= A(i,2);
        Bmerged(rBmerged,3) = A(i,3);
        %Bmerged(rBmerged,4) = A(i,4);
        %Bmerged(rBmerged,5) = A(i,5);
        rBmerged = rBmerged+1;
    else
        BNotmerged(rBNotmerged,1) = A(i,1);
        BNotmerged(rBNotmerged,2)= A(i,2);
        BNotmerged(rBNotmerged,3) = A(i,3);
        %BNotmerged(rBNotmerged,4) = A(i,4);
        %BNotmerged(rBNotmerged,5) = A(i,5);
        rBNotmerged = rBNotmerged+1;
    end
    
end
%hist(Bmerged(:,col),200);
[f,xi, bw] = ksdensity(Bmerged(:,col)); %to change
disp('bandwidth matlab kernel for likelihood Pmerged');
disp(bw);

[fnot,xinot, bwnot] = ksdensity(BNotmerged(:,col)); %to change
disp('bandwidth matlab kernel for likelihood PmergedNOT');
disp(bwnot);
%hist(BNotmerged(:,col),200);

figure
%subplot(2,2,4)       
yyaxis left          
plot(xinot,fnot)           
yyaxis right         
plot(xi,f)
title('Probabilities')
%plot(xinot,fnot);
disp('finished');