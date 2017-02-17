
for k=1:10

filename = sprintf('cnninput/x50_%d/x50.txt',k);  
data = importdata(filename);    

for i=1:size(data,1)

 img1 = reshape(data(i,:),250,500);
 img2 = imresize(img1,0.5);
 if i==1
     new = img2(:)';
 else
     new = [new; img2(:)'];
 end
 
end

filename = sprintf('cnninput/x50_%d/x50b.txt',k);  
dlmwrite(filename,new);
end