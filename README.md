# EE285-Team-Calexy
A cunstomized combination of age/race/gender classifier based on VGG16 and human face detector based on YOLOv3.

## Installing The Base darknet
First clone our git repository here. This can be completed by:
```
git clone https://github.com/MichaelHanJ/EE285-Team-Calexy.git
cd EE285-Team-Calexy
make clean
make
```
We assume that you are using GPUs to run this program, but if you want to run only with CPUs, you need to make some revisions on `Makefile`, where you need to change the first line into:
```
GPU = 0
```
If you want to compile with OpenCV, you should change the 2nd line of the 'Makefile' to read:
```
OPENCV=1
```
After you complete all the changes, you can refresh your Makefile by:
```
make clean
make
```
If this works, you will see a whole bunch of compiling information fly by:
```
mkdir -p obj
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast....
.....
gcc -I/usr/local/cuda/include/  -Wall -Wfatal-errors  -Ofast -lm....
```
If you have any errors, try to fix them according to given hints. If everything seems to have complied correctly, you can try to run it by:
```
./darknet
```
You should get the output:
```
usage: ./darknet <function>
```
Now, you are ready to play with this custimoized YOLOv3 network for human face detection.
## Detection Using A Pre-Trained Model
We have trained YOLOv3 and YOLOv3 models. You will have to download the pre-trained weight file [here](https://drive.google.com/file/d/1wDD2I4vNO7U5FDoXKz9JM8P8xf498kwz/view?usp=sharing) for YOLOv3 model. Or just run this to get the pre-trained weights saved in Google Drive.
```
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
gdrive_download 1wDD2I4vNO7U5FDoXKz9JM8P8xf498kwz yolo-obj.weights
```
Please make sure the weight file is stored in the EE285-Team-Calexy folder.
Then run the detector!
```
./darknet detector test testexample/test_1.JPEG cfg/yolo-obj.cfg yolo-obj.weights
```
You will see some outputs like this:
```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32  0.299 BFLOPs
    1 conv     64  3 x 3 / 2   416 x 416 x  32   ->   208 x 208 x  64  1.595 BFLOPs
    2 conv     32  1 x 1 / 1   208 x 208 x  64   ->   208 x 208 x  32  0.177 BFLOPs
    ......
    104 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256  1.595 BFLOPs
    105 conv     18  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x  18  0.025 BFLOPs
    106 yolo
Loading weights from yolo-obj.weights...Done!
Enter Image Path: 
```
You need to enter image path again:
```
Enter Image Path: testexample/test_1.JPEG
```
You will get some outputs like:
```
testexample/test_1.JPEG: Predicted in 0.419771 seconds.
face: 100%
43 22 127 130
Image Path saved in cropped_images/image_path.txt
Enter Image Path: 
```
This network prints out the objects detected, the confidence, cooridinates for box and how long it took to find them. The detected result with bounding box is saved to predictions.png in the EE285-Team-Calexy folder.
If you want to exit the test process, you can press `Control + C` to exit.

