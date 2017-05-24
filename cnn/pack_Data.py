from numpy import genfromtxt
import gzip, cPickle

dirg = 'run2/' 

name1 = dirg+'train1.txt'
train_set= genfromtxt(name1)
name = dirg+'train1.pkl.gz' 
f = gzip.open(name,'wb')
cPickle.dump([train_set], f, protocol=2)
f.close()
del train_set

name1 = dirg+'val1.txt'
valid_set= genfromtxt(name1)
name = dirg+'val1.pkl.gz' 
f = gzip.open(name,'wb')
cPickle.dump([val_set], f, protocol=2)
f.close()
del val_set

name1 = dirg+'test1.txt'
test_set= genfromtxt(name1)
name = dirg+'test1.pkl.gz' 
f = gzip.open(name,'wb')
cPickle.dump([test_set], f, protocol=2)
f.close()
del test_set