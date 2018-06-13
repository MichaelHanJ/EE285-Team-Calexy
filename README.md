# EE285-Team-Calexy
A cunstomized combination of age/race/gender classifier based on VGG16 and human face detector based on YOLOv3.

## Installing The Base darkNet
First clone our git repository here. This can be completed by:
```
git clone https://github.com/MichaelHanJ/EE285-Team-Calexy.git
cd EE285-Team-Calexy
```
We assume that you are using GPU to run this program, but if you want to run only with CPU, you need to make some revisions on `Makefile`, where you need to change the first line into:
```
GPU = 0
```
make clean
make
```
If this works, you will see a bunch of compiling information fly like:
```
mkdir -p obj
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
.....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast -lm....
```
