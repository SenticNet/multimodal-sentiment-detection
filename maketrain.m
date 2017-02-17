for k=1:10
    dataallx = [];
    dataally = [];
    
    for j=1:10 
       
      if k~=j
         
          filename = sprintf('cnninput/x50_%d/x50b.txt',j);
          trainx = importdata(filename);
          filename = sprintf('cnninput/x50_%d/y50.txt',j);
          trainy = importdata(filename);
          
          if size(dataallx,1)<2
            dataallx = [trainx];
            dataally = [trainy];  
          else    
            dataallx = [dataallx;trainx];
            dataally = [dataally;trainy];
          end
          
      end
        
        
    end
    
    n1 = floor(0.7*size(dataallx,1));
    
    trainx = dataallx(1:n1,:);
    trainy = dataally(1:n1,:);
    
    valx = dataallx(n1+1:end,:);
    valy = dataally(n1+1:end,:);
       
    filename = sprintf('cnninput/x50_%d/trainx.txt',k);
    dlmwrite(filename,trainx);
    filename = sprintf('cnninput/x50_%d/trainy.txt',k);
    dlmwrite(filename,trainy);
    filename = sprintf('cnninput/x50_%d/valx.txt',k);
    dlmwrite(filename,valx);
    filename = sprintf('cnninput/x50_%d/valy.txt',k);
    dlmwrite(filename,valy);
    
end
