#ifndef REGION_H
#define REGION_H

#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

struct Coord{
	int row;
	int col;

	Coord(int r,int c){
		row=r;
		col=c;
	}

	bool operator==(const Coord &obj){
		return (row == obj.row && col == obj.col);
	}
};

class Region{

	std::vector<Coord> pixels;
	std::vector<Coord> boundaryPixels;	
	//float meanValue;

	cv::Mat *image;
	cv::Mat *gray_image;
	cv::Mat labImage;
	//IplImage *image;

public:

	

	float calculateGLColorMeanValue(int channel){
		//cv::Mat labImage = image->clone();

		//cv::cvtColor(*image, labImage, CV_BGR2Lab);

		float mean=0;

		for(int i=0;i<pixels.size();i++){
					cv::Vec3b labVector = labImage.at<cv::Vec3b>(pixels.at(i).row,pixels.at(i).col);
					//cout << "val[channel]:		"<< labVector.val[channel]<< "	channel:		"<< channel << endl;
					mean += labVector.val[channel];

				}
		//cout << "mean value:		"<< mean/pixels.size()<< "	channel:		"<< channel +1 << endl;
		return (mean/pixels.size());
	}


	float calculateGLTexture () {

		float stddev = 0;                              //define variance
		//float mean = calculateGLColorMeanValue();        //call mean value function
		vector<int> histogram;							 //define histogram
		float numPixels = getNoOfPixels();
		//calculate mean
		float mean=0;
		for(int i=0;i<pixels.size();i++)
			mean+=gray_image->at<uchar>(pixels.at(i).row,pixels.at(i).col);

				//meanValue=sum/pixels.size();

		mean /= pixels.size();
		//set histogram entries to zero
		for (int i=0; i<256; i++)
			histogram.push_back(0);

		//calculate histogram of the region
		for (int i=0; i< numPixels; i++){
			Coord pixel = getAt(i);
			histogram[gray_image->at<uchar>(pixel.row,pixel.col)] +=1;
		}    

		//calculate standard deviation
		for ( int i =0; i<256; i++){
			stddev += sqrt(pow((i - mean),2)*histogram[i]/numPixels);
		}

		return stddev;
	}

	float calculateEntropy(){
		vector<int> histogram;
		float numPixels = getNoOfPixels();
		float entropy=0;

		//set histogram entries to zero
		for (int i=0; i<256; i++)
		histogram.push_back(0);

		//calculate histogram of the region
		for (int i=0; i< numPixels; i++){
			Coord pixel = getAt(i);
			histogram[gray_image->at<uchar>(pixel.row,pixel.col)] +=1;
			}
		//calculate entropy
		for ( int i =0; i<256; i++){
			if(histogram[i]/numPixels == 0){
				entropy +=0;
			}
			else{
			entropy += -(histogram[i]/numPixels*log2(histogram[i]/numPixels));
			}

		}
		if (entropy >8){
			cout << "entropy is higher than 8:	" << entropy <<endl;
		}
	return entropy;

	}



//public:
	Region(){
		
	}

	Region(cv::Mat *img,cv::Mat *g_img){
	//Region(IplImage *img){	
		image=img;
		gray_image=g_img;
		labImage = image->clone();
		 cv::cvtColor(*image, labImage, CV_BGR2Lab);
	}

	void addPixel(Coord p){
		pixels.push_back(p);
	}

	/*void addPixelAndRefresh(Coord p){
		pixels.push_back(p);
		refreshMeanValue();
	}*/

	void addBoundaryPixel(Coord p){
		boundaryPixels.push_back(p);
	}

	bool findBoundaryPixel(Coord p){
		for(int i=0;i<boundaryPixels.size();i++)
			if(boundaryPixels[i]==p)
				return true;
		return false;
	}

	Coord getAt(int pos){
		return pixels.at(pos);
	}

	/*float getMeanValue(){
		return meanValue;
	}*/

	int getNoOfPixels(){
		return pixels.size();
	}

	int getNoOfBoundaryPixels(){
		return boundaryPixels.size();
	}

	Coord getBoundaryPixel(int i){
		return boundaryPixels[i];
	}

	void adjustBoundaryPixels(const vector<vector<int>> &regionsLabels){
		const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
		const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

		for (int i = 0; i < getNoOfBoundaryPixels(); i++) {
			int nr_p = 0;
			Coord p=getBoundaryPixel(i);
			for (int k = 0; k < 8; k++) {
				int x = p.col + dx8[k], y = p.row + dy8[k];
				
				if (x >= 0 && x < regionsLabels.size() && y >= 0 && y < regionsLabels[0].size()) 
					if (regionsLabels[p.col][p.row] != regionsLabels[x][y]) 
						nr_p += 1;				
			}

			if (nr_p <2){
				boundaryPixels.erase(boundaryPixels.begin()+i);
				i--;
			}

		}
	}



};

#endif
