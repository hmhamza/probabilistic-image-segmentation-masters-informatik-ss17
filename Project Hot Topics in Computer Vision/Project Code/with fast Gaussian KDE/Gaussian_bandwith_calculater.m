path = 'Final.txt';         %define path of training data file
A = textread((path));
col =1;                     %defines the bandwith to calculate 1: color, 2: texture, 3: arrangement
maxValue = max(A(:,col));
minValue = min(A(:,col));


%normalize data
for i=1:length(A)
 A(i,1)=(A(i,1) -  min(A(:,1)))/(abs(min(A(:,1))-max(A(:,1))));
 A(i,2)=(A(i,2) -  min(A(:,2)))/(abs(min(A(:,2))-max(A(:,2))));
 A(i,3)=(A(i,3) -  min(A(:,3)))/(abs(min(A(:,3))-max(A(:,3))));
end

Bmerged = zeros(1,1);
rBmerged = 1;
BNotmerged = zeros(1,1);
rBNotmerged = 1;
for i=1:size(A,1)   %go over rows
    if A(i,4)==1
        Bmerged(rBmerged,1) = A(i,1);
        Bmerged(rBmerged,2)= A(i,2);
        Bmerged(rBmerged,3) = A(i,3);
        
        rBmerged = rBmerged+1;
    else
        BNotmerged(rBNotmerged,1) = A(i,1);
        BNotmerged(rBNotmerged,2)= A(i,2);
        BNotmerged(rBNotmerged,3) = A(i,3);
 
        rBNotmerged = rBNotmerged+1;
    end
    
end



[f,xi, h] = ksdensity(Bmerged(:,col));
disp('h scaled');
disp(h)
[fnot,xinot, bwnot] = ksdensity(BNotmerged(:,col));
disp('hNOT scaled');
disp(bwnot);


disp('finished')