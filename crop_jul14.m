clear
tic
load moud.txt;

filePath=sprintf('cnninput');
if( exist(filePath, 'dir') )
    rmdir( filePath, 's' );
end

pause(10);
mkdir(filePath);

F = dir('F:\\MOUD\\MOUD\\VideoReviews\\transcriptions2\\*.csv');

for k1=1:10
    
 indexcnt = 1;
 %for ii=1:length(F)
 startk1 = 1+50*(k1-1);
 endk1 = 50*k1;
 if k1==10
     endk1 = 498;
 end
 dataall = [];
 datayall = [];
 dataind = [];
 
 for ii=startk1:endk1 

 if moud(ii) ~= 0 && ii ~=116
     
 if moud(ii) == -1
         moud(ii)=0;
 end   

 filename = sprintf('CLM2\\%s',F(ii).name);
 facexy = importdata(filename);
 facexy = floor(facexy);
 facexy2 = facexy(1,:);
 
 for nr=1:10:size(facexy,1)
     facexy2=[facexy2;facexy(nr,:)];  
 end

 facexy = facexy2;    

 filename = sprintf('F:\\MOUD\\MOUD\\VideoReviews\\transcriptions2\\%s',F(ii).name);
 dataf=importdata(filename);
 
 dataf2 = dataf(1,:);
 facexy2 = facexy(1,:);
 
 for nr=1:10:size(facexy,1)
    
     dataf2 = [dataf2;dataf(nr,:)];
     facexy2=[facexy2;facexy(nr,:)];
     
 end
 
 dataf = dataf2;
 facexy = facexy2;

 miny=min(facexy(:,2:69)')';
 minx=min(facexy(:,70:end)')';
 maxy=max(facexy(:,2:69)')';
 maxx=max(facexy(:,70:end)')';
 
 maxpix = 250;

 for j=1:size(facexy,1)
    
       dataf2 = reshape(dataf(j,:),288,352);
       dataf2d = zeros(2*maxpix,2*maxpix);
       dataf2a = zeros(maxpix,maxpix);
       if minx(j)<=0
           minx(j) = 1;
       end
       if miny(j)<=0
           miny(j) = 1;
       end
       if maxx(j) >= 288
           maxx(j) = 288;
       end
       if maxy(j) >= 352
           maxy(j) = 352;
       end
       
       dataf2d(1:maxx(j)-minx(j)+1,1:maxy(j)-miny(j)+1) = dataf2(minx(j):maxx(j),miny(j):maxy(j));
       dataf2a = dataf2d(1:maxpix,1:maxpix);
       
       if j==1
           old = dataf2a;
       else
         dataf2c = [dataf2a old];
         old = dataf2a;
         dataf2b = dataf2c(:)';
         if j==2
           dataf3 = dataf2b(:)';
         else    
           dataf3 = [dataf3;dataf2b(:)'];
         end
       end        
 end
 
   dataall{indexcnt}=mat2cell(dataf3);
   datayall{indexcnt}=mat2cell(ones(1,size(dataf3,1))*moud(ii));
   dataind{indexcnt}=mat2cell(ii*ones(1,size(dataf3,1)));
   indexcnt = indexcnt+1;
 end 
 
 end

 for ii=1:indexcnt-1
    
     if ii==1
         dataall2 =cell2mat(dataall{ii});
         datayall2 = cell2mat(datayall{ii});
         dataind2 = cell2mat(dataind{ii});
     else
         dataall2 = [dataall2; cell2mat(dataall{ii})];
         datayall2 = [datayall2 cell2mat(datayall{ii})];
         dataind2 = [dataind2 cell2mat(dataind{ii})];
     end    
     
 end

 filePath=sprintf('cnninput\\x50_%d',k1);
 if( exist(filePath, 'dir') )
    rmdir( filePath, 's' );
 end

 pause(10);
 mkdir(filePath);
 
 
 filename = sprintf('%s\\x50.txt',filePath);
 dlmwrite(filename,dataall2);
 filename = sprintf('%s\\y50.txt',filePath);
 dlmwrite(filename,datayall2');
 filename = sprintf('%s\\ind.txt',filePath);
 dlmwrite(filename,dataind2');
 

 end
 toc