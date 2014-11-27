n1='www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/train_images.txt'
n2='www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/test_images.txt'
n3='www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/train_labels.txt'
n4='www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/test_labels.txt'
n5='www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/faithful.txt'

mkdir data
pushd .
cd data
wget $n1 
wget $n2 
wget $n3 
wget $n4 
wget $n5 
popd
