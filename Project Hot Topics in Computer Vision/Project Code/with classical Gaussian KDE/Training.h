#ifndef TRAINING_H
#define TRAINING_H

#include<iostream>
#include<fstream>
#include <opencv2/opencv.hpp>

#include "Region.h"
#include "slic.h"
#include "Structs.h"
#include"Timer.h"

using namespace std;
using namespace cv;

class Training{

	IplImage *img;
	Mat image,gray_image;

	vector<vector<int>> segmentsData;					//Storing data from file
	int noOfSegments;
	vector <Region> regions;							//Storing regions from SLIC	
	vector<vector<int>> regionsLabels;					//Regions labels from SLIC
	vector<int> regionsCounter;
	vector<vector<OneRegion>> adjacentRegions;
	int noOfRegions;

	/* Calculates arrangement of 2 regions */
	float calculateArrangement(int reg1,int reg2){		
		const int neighborX[4] = {-1, 0,  0,  1};
		const int neighborY[4] = { 0, -1, 1, 0};

		float count=0;
		for(int i=0;i<regions[reg1].getNoOfBoundaryPixels();i++){

			Coord p=regions[reg1].getBoundaryPixel(i);
			for (int k = 0; k < 4; k++) {	
				int x = p.col + neighborX[k], y = p.row + neighborY[k];
				Coord temp(y,x);

				if(regions[reg2].findBoundaryPixel(temp)){
					count++;
				}
			}
		}

		float totalBoundaryPixels=regions[reg1].getNoOfBoundaryPixels()+regions[reg2].getNoOfBoundaryPixels();

		return (count/totalBoundaryPixels);
	}

	/* This function checks whether the 2 regions sent as parameters belong to one same region in the original segmentation */
	bool Check(Region *reg1,Region *reg2){
		vector<int> reg1Counter,reg2Counter;
		for(int i=0;i<noOfSegments;i++){
			reg1Counter.push_back(0);
			reg2Counter.push_back(0);		
		}


		for(int i=0;i<reg1->getNoOfPixels();i++){
			int row=reg1->getAt(i).row,col=reg1->getAt(i).col;
			int segmentNo=segmentsData[row][col];
			reg1Counter[segmentNo]++;
		}
		int max1=0;
		for(int i=1;i<reg1Counter.size();i++){
			if(reg1Counter[i]>reg1Counter[max1])
				max1=i;
		}
		float percent1=((float)reg1Counter[max1]/(float)reg1->getNoOfPixels())*100;

		for(int i=0;i<reg2->getNoOfPixels();i++){
			int row=reg2->getAt(i).row,col=reg2->getAt(i).col;
			int segmentNo=segmentsData[row][col];
			reg2Counter[segmentNo]++;
		}
		int max2=0;
		for(int i=1;i<reg2Counter.size();i++){
			if(reg2Counter[i]>reg2Counter[max2])
				max2=i;
		}
		float percent2=((float)reg2Counter[max2]/(float)reg2->getNoOfPixels())*100;

		return (max1==max2);
	}

	/* Merges 2 regions */
	void Merge(int r1,int r2){

		Region *reg1=&regions[r1];
		Region *reg2=&regions[r2];

		int curr=regionsLabels[reg1->getAt(0).col][reg1->getAt(0).row];
		for(int i=0;i<reg2->getNoOfPixels();i++){
			Coord pixel=reg2->getAt(i);
			reg1->addPixel(pixel);
			regionsLabels[pixel.col][pixel.row]=curr;
		}

		for(int i=0;i<reg2->getNoOfBoundaryPixels();i++)
			reg1->addBoundaryPixel(reg2->getBoundaryPixel(i));

		reg1->adjustBoundaryPixels(regionsLabels);

		for(int i=0;i<regionsCounter.size();i++){
			if(regionsCounter[i]==r2)
				regionsCounter[i]=r1;
		}

		noOfRegions--;
	}

	/* Displays contours of the current regions */
	void displayContours(char *name){
		IplImage *contourImage = cvCloneImage(img);
		for (int i = 0; i < regions.size(); i++) {
			if(regionsCounter[i]==i){
				for (int j = 0; j < regions[i].getNoOfBoundaryPixels(); j++) {
					Coord pixel=regions[i].getBoundaryPixel(j);
					int row=pixel.row, col=pixel.col;
					cvSet2D(contourImage, row,col, CV_RGB(255,0,0));	
				}
			}
		}
		cvShowImage(name, contourImage);
	}

	/* Loads training data from data file. This data file has been downloaded from Berkeleys Image data set */
	void LoadTrainingData(string infile){

		ifstream fin(infile);
		string input;
		int width,height,segment,row,col1,col2;		

		getline(fin,input);getline(fin,input);getline(fin,input);getline(fin,input);		//Ignoring first 4 line

		fin>>input;
		fin>>width;
		fin>>input;
		fin>>height;
		fin>>input;
		fin>>noOfSegments;

		for(int i=0;i<height;i++){
			vector<int> temp;
			for(int j=0;j<width;j++)
				temp.push_back(-1);
			segmentsData.push_back(temp);
		}

		fin.ignore();
		getline(fin,input);getline(fin,input);getline(fin,input);getline(fin,input);	//Ignoring next 4 lines

		while(!fin.eof()){
			fin>>segment;
			fin>>row;
			fin>>col1;
			fin>>col2;

			for(int i=col1;i<=col2;i++)
				segmentsData[row][i]=segment;			
		}
	}

	/* Running SLIC on the image. The SLIC results are got from the algorithm and stored in vectors */
	void RunSLIC(const char* imagePath,int n,int nc){
		cout<<"\nRunning SLIC...\t\t\t\t";

		image=imread(imagePath,CV_LOAD_IMAGE_UNCHANGED);
		cvtColor( image, gray_image, COLOR_BGR2GRAY );

		img = cvLoadImage(imagePath, 1);
		IplImage *lab_image = cvCloneImage(img);
		cvCvtColor(img, lab_image, CV_BGR2Lab);

		int w = img->width, h = img->height,noOfSuperpixels;
		noOfSuperpixels=(w*h)/n;
		double step = sqrt((w * h) / (double) noOfSuperpixels);

		Slic slic;
		slic.generate_superpixels(lab_image, step, nc);
		slic.create_connectivity(lab_image);

		for(int i=0;i<noOfSuperpixels+10000;i++){			
			Region newRegion(&image,&gray_image);
			regions.push_back(newRegion);
		}
		
		slic.getResults(img,regions,regionsLabels);
		noOfRegions=regions.size();
		for(int i=0;i<regions.size();i++){
			regionsCounter.push_back(i);
		}

		
	}

	void FindAdjacentRegions(string adjacentRegionsFile){
		cout<<"\nFinding Adjacent Regions...\t\t";
		
		vector<vector<bool>> adjacentRegionsCheck;
		for(int i=0;i<regions.size();i++){
			vector<bool> temp;
			for(int j=0;j<regions.size();j++){
				temp.push_back(false);
			}
			adjacentRegionsCheck.push_back(temp);
		}				

		for(int i=0;i<regions.size();i++){
			vector<OneRegion> temp;
			adjacentRegions.push_back(temp);
		}		

		double x=regions.size();
		double x2=-4.96281576*pow(10,-5)*pow(x,2);
		double x1=1.170450583*0.1*x;
		int y=x2+x1 + 10.32622578;
		y=y+10;

		ofstream fout(adjacentRegionsFile);

		for(int i=0;i<regions.size();i++){
			int j=i-y,k=i+y;
			if(j<0)
				j=0;
			if(k>regions.size())
				k=regions.size();

			for(;j<k;j++){
				if(i!=j){
					float arg=calculateArrangement(i,j);
					if(arg>0){
						adjacentRegions[i].push_back(OneRegion(j,arg));
						fout<<i<<" "<<j<<" "<<arg<<endl;
						adjacentRegionsCheck[i][j]=true;
						adjacentRegionsCheck[j][i]=true;
					}
				}
			}
		}
	}

	void ReadAdjacentRegionsFromFile(string adjacentRegionsFile){
		cout<<"\nReading Adjacent Regions From File...   ";

		for(int i=0;i<regions.size();i++){
			vector<OneRegion> temp;
			adjacentRegions.push_back(temp);
		}		

		ifstream fin(adjacentRegionsFile);

		int r1,r2;
		float a;
		while(!fin.eof()){
			fin>>r1;
			fin>>r2;
			fin>>a;
			adjacentRegions[r1].push_back(OneRegion(r2,a));
		}

	}

	void Perform(string outfile){
		cout<<"\nPerforming...\t\t\t\t";
		ofstream fout(outfile);

		bool anyMerge=true;
		for(int a=1;anyMerge;a++){
			anyMerge=false;

			for(int i=0;i<adjacentRegions.size();i++){
				int actualRegion=regionsCounter[i];
				for(int j=0;j<adjacentRegions[i].size();j++){					
					int adjacentReg=adjacentRegions[i][j].region;					
					adjacentReg=regionsCounter[adjacentReg];

					if(actualRegion!=adjacentReg){
						Region *region1=&regions[actualRegion],*region2=&regions[adjacentReg];
						bool whetherMerge=Check(region1,region2);

						float colorDiff=sqrt( pow((region1->calculateGLColorMeanValue(0) -region2->calculateGLColorMeanValue(0)),2)+
											 pow((region1->calculateGLColorMeanValue(1) -region2->calculateGLColorMeanValue(1)),2)+
											 pow((region1->calculateGLColorMeanValue(2) -region2->calculateGLColorMeanValue(2)),2));
						

						float textureDiff=abs(region1->calculateGLTexture()-region2->calculateGLTexture());
						fout<<colorDiff<<" "<<textureDiff<<" "<<adjacentRegions[i][j].arrangement<<" "<<whetherMerge<<endl;	//CHECK THIS  [actualRegion][random]

						if(whetherMerge){
							anyMerge=true;
							Merge(actualRegion,adjacentReg);
						}
					}
				}
			}
		}
	}

	void SaveResult(const char *resultImagePath){
		IplImage* resultImg=cvCloneImage(img);
		
		for (int i = 0; i < regions.size(); i++) {
			if(regionsCounter[i]==i){
				for (int j = 0; j < regions[i].getNoOfBoundaryPixels(); j++) {
					Coord pixel=regions[i].getBoundaryPixel(j);
					int row=pixel.row, col=pixel.col;
					cvSet2D(resultImg, row,col, CV_RGB(255,0,0));	
				}
			}
		}		
		cvSaveImage(resultImagePath,resultImg);
	}

	void ClearData(){
		segmentsData.clear();
		regions.clear();
		regionsLabels.clear();
		regionsCounter.clear();
		adjacentRegions.clear();
		
	}

public:
	Training(){	}

	/* The main driver function of the class */
	void Start(string name,string trainingDataFile,const char* imagePath,int noOfPixels,int nc,string adjacentRegionsFile,bool isAdjacentFile,string statsFile,const char* resultImagePath){

		cout<<"\t\t\t  * Training "<<name<<" *\n\n";

		Timer totalTime,localTime;
		totalTime.Start();

		LoadTrainingData(trainingDataFile);
		
		localTime.Start();
		RunSLIC(imagePath,noOfPixels,nc);
		localTime.LocalEnd();
		cout<<"   Superpixels: "<<regions.size();
		
		localTime.Start();
		if(isAdjacentFile)
			ReadAdjacentRegionsFromFile(adjacentRegionsFile);
		else
			FindAdjacentRegions(adjacentRegionsFile);		
		localTime.LocalEnd();
				
		localTime.Start();
		Perform(statsFile);
		localTime.LocalEnd();
		
		SaveResult(resultImagePath);

		ClearData();	

		totalTime.TotalEnd();

	}

	/* This function is ised by other classes that want to get the already processed training data */
	vector<Stat> GetTrainingData(string filename){
		ifstream fin(filename);

		vector<Stat> data;
		float color,texture,arrangement;
		bool isMerged;

		while(!fin.eof()){
			fin>>color;
			fin>>texture;
			fin>>arrangement;
			fin>>isMerged;

			data.push_back(Stat(color,texture,arrangement,isMerged));
		}

		return data;
	}

};


#endif
