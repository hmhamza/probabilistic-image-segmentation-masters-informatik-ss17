{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\froman\fcharset0 TimesNewRomanPSMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;}
{\*\expandedcolortbl;;\csgenericrgb\c0\c0\c0;\csgenericrgb\c100000\c100000\c100000;}
\margl1440\margr1440\vieww12560\viewh12080\viewkind0
\deftab708
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\qc\partightenfactor0

\f0\b\fs36 \cf0 \ul \ulc0 Training\
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\partightenfactor0

\b0\fs24 \cf0 \ulnone In Main.cpp, there is a function call 
\i\b doTraining()
\i0\b0 . If you want to do training, call this function from 
\i\b main()
\i0\b0 . There is a vector 
\i\b vector <string> trainingFiles
\i0\b0 . Add all the file names in this vector that you want to do training with. All of the training files (data files as well as image files) should be placed in the folder 
\b Training/New
\b0  and the output files will be generated in the same folder. After training the new bandwidths for the probability estimator can be calculated by the Matlab code \'84getbandwidth.mr\'93.\
\
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\qc\partightenfactor0

\b\fs36 \cf0 \ul Testing\
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\partightenfactor0

\b0\fs24 \cf0 \ulnone In Main.cpp, there is a function call 
\i\b doTesting()
\i0\b0 . If you want to do testing, call this function from 
\i\b main()
\i0\b0 . There is a vector 
\i\b vector <string> testingFiles
\i0\b0 . Add all the file names in this vector that you want to do training with. The training images will be in the folder 
\i\b Testing/Images
\i0\b0 , results will be generated in 
\b Testing/Results
\b0  and boundary maps will be generated in 
\b Testing/Boundary Maps
\b0 .\
\
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\qc\partightenfactor0

\b\fs36 \cf0 \ul Compiling\
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\partightenfactor0

\b0\fs24 \cf0 \ulnone The software can be compiled with the g++ compiler.\
Example:\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardeftab708\pardirnatural\partightenfactor0
\cf2 \cb3 \CocoaLigature0 g++ -I/usr/local/include -fopenmp -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_core Main.cpp Region.h RegionGrowing.h slic.cpp slic.h Structs.h Training.h Timer.h -o final\cf0 \cb1 \CocoaLigature1 \
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\qc\partightenfactor0

\b\fs36 \cf0 \ul \
Code description\
\pard\pardeftab708\ri-46\sl276\slmult1\sa200\partightenfactor0

\b0\fs24 \cf0 \ulnone A detailed code description can be found in the root directory of the code.
\b\fs36 \ul \
}