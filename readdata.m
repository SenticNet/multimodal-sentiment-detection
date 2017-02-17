

F = dir('videos\*.mp4');
for ii = 2:length(F)
    
[pathi,name,ext] = fileparts(F(ii).name);

filename = sprintf('videos\\%s',F(ii).name);
obj = mmreader(filename);
filename = sprintf('transcriptions3\\%s.csv',name);
%filename = 'VideoReviews\transcriptions2\100_makeup.csv'
if exist(filename, 'file') == 2

time = importdata(filename);
time = time.data;
data = obj.read();

[x,y,n,f]=size(data);
data2 = zeros(x*y,1);

for i=1:f
    a=rgb2gray(data(:,:,:,i));
    
    b = a(:);
    data2 = [data2 b];
    
end
data2 = data2(:,2:end);
data2 = data2';
[n1 n2] = size(time);
for i=1:n1
   startf = floor(time(i,1)*obj.FrameRate);
   endf = floor(time(i,2)*obj.FrameRate); 
   if startf==0
       startf=1;
   end
   if endf > f
       endf = f;
   end
   data3 = data2(startf:endf,:);
   filename = sprintf('transcriptions2\\%s_%d.csv',name,i);
%   filename = sprintf('VideoReviews\transcriptions2\100_makeup_%d.csv',i); 
   dlmwrite(filename,data3);
end

end
%data2 = data2(:,2:end);
%dlmwrite('100_makeup.txt',data2);
end