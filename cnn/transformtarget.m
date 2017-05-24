function T2=transformtarget(T,num)
%num = 2;
T2 = zeros(1,num);

for t=1:length(T)
    x = zeros(1,num);
    x(T(t))=1;
    T2 = [T2;x];
end

T2 = T2(2:end,:);
end