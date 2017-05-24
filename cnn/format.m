clear

for k=0:9

filename = sprintf('run2/train_cnn%d',k);  
layer3_mat = load(filename); 

[n1 n2] = size(layer3_mat);

w = 20; f=0; check = 0;
f=reshape(layer3_mat,w,n1/w);

filename = sprintf('../cnninput/x50_%d/trainy.txt',k+1);
%filename = sprintf('run2/train%d_y',k);  
%load 'run2/train0_y';
train0_y=load(filename);

label_y = train0_y;

[n1 n2]=size(f);
[n3 n4]=size(label_y);

if n2 < n3
label_y = label_y(1:n2,:);
else
f = f(:,1:n3);
end

g = [f; label_y'+1];

filename = sprintf('set_train_rnn%d',k);
dlmwrite(filename,g);

filename = sprintf('run2/test_cnn%d',k);  
layer3_mat = load(filename); 

[n1 n2] = size(layer3_mat);

w = 20; f=0; check = 0;
f=reshape(layer3_mat,w,n1/w);


filename = sprintf('../cnninput/x50_%d/y50.txt',k+1);
%filename = sprintf('run2/test%d_y',k); 
%load 'run2/test0_y';
test0_y=load(filename);

label_y = test0_y;

[n1 n2]=size(f);
[n3 n4]=size(label_y);

if n2 < n3
label_y = label_y(1:n2,:);
else
f = f(:,1:n3);
end

g = [f; label_y'+1];

filename = sprintf('set_test_rnn%d',k);
dlmwrite(filename,g);

filename = sprintf('run2/val_cnn%d',k);  
layer3_mat = load(filename); 

[n1 n2] = size(layer3_mat);

w = 20; f=0; check = 0;
f=reshape(layer3_mat,w,n1/w);

filename = sprintf('../cnninput/x50_%d/valy.txt',k+1);
%filename = sprintf('run2/val%d_y',k);
%load 'run2/val0_y';
val0_y=load(filename);

label_y = val0_y;

[n1 n2]=size(f);
[n3 n4]=size(label_y);

if n2 < n3
label_y = label_y(1:n2,:);
else
f = f(:,1:n3);
end


g = [f; label_y'+1];

filename = sprintf('set_val_rnn%d',k);
dlmwrite(filename,g);

end