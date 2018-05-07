#ifndef REGIONGROWING_H
#define REGIONGROWING_H

#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>

#include"Training.h"
#include"GroundTruth.h"
#include <omp.h>
using namespace std;
using namespace cv;

class RegionGrowing{

	IplImage *img;

	Mat image,gray_image;
	vector <Region> regions;
	vector<vector<int>> regionsLabels;

	vector<int> regionsCounter;
	vector<vector<OneRegion>> adjacentRegions;
	vector<Stat> trainingData;
	vector<float> Priors;
	int noOfRegions;

	vector<float> colorTDM,colorTDNM,textureTDM,textureTDNM,arrangementTDM,arrangementTDNM;		//TDM: Training Data Merged		TDNM: Training Data Not Merged

	Mat checkMatrix;

	
	float calculateArrangement(int reg1,int reg2){


		const int neighborX[4] = {-1, 0,  0,  1};
		const int neighborY[4] = { 0, -1, 1, 0};

		float count=0;
		for(int i=0;i<regions[reg1].getNoOfBoundaryPixels();i++){
			bool flag=false;
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

	void displayBoundaryOfaRegion(int i,char *name){
		IplImage *contourImage = cvCloneImage(img);
		for (int j = 0; j < regions[i].getNoOfBoundaryPixels(); j++) {
			Coord pixel=regions[i].getBoundaryPixel(j);
			int row=pixel.row, col=pixel.col;
			cvSet2D(contourImage, row,col, CV_RGB(255,0,0));	
		}


		cvShowImage(name,contourImage);


	}

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

	

	int Probability(int reg) {
		float maxProb =-1;
		int actualRegion=regionsCounter[reg];



		Region *current=&regions[actualRegion];
		//define bandwidth of kernels 
		float hColor = 3.4078, hColorNOT = 4.4450;
		float hText = 39.8589, hTextNOT = 30.7168;
		float hArr= 0.0104, hArrNOT = 0.0047;

	

		int maxProbRegion=0;
		for(int i=0;i<adjacentRegions[reg].size();i++){
			int adjacentReg=adjacentRegions[reg][i].region;
			adjacentReg=regionsCounter[adjacentReg];

			if(actualRegion!=adjacentReg){



				Region *adjacent= &regions[adjacentReg];

				float Diffcolor=sqrt(pow((current->calculateGLColorMeanValue(0)- adjacent->calculateGLColorMeanValue(0)),2)+
					pow((current->calculateGLColorMeanValue(1)- adjacent->calculateGLColorMeanValue(1)),2)+
					pow((current->calculateGLColorMeanValue(2)- adjacent->calculateGLColorMeanValue(2)),2));
				float DiffTexture=abs(current->calculateGLTexture()-adjacent->calculateGLTexture());
				float Arrangement=adjacentRegions[reg][i].arrangement;

				float result=0; //PmergeNOT,, Ptexture, Parrangement;
				vector<float> LikeColor, LikeTexture, LikeArrangement, LikeEntroy;


#pragma omp parallel sections
				{		
			#pragma omp section
					{				
				LikeColor = likelihoods(Diffcolor, 0, hColor, hColorNOT);
					}
			#pragma omp section
					{
				LikeTexture = likelihoods(DiffTexture, 1, hText, hTextNOT);
					}
			#pragma omp section
					{
				LikeArrangement = likelihoods(Arrangement, 2, hArr, hArrNOT);
					}
				}


				result = (LikeColor[0]*LikeTexture[0]*LikeArrangement[0]*Priors[0])/
					((LikeColor[0]*LikeTexture[0]*LikeArrangement[0]*Priors[0])+(LikeColor[1]*LikeTexture[1]*LikeArrangement[1]*Priors[1]));

				

				
				if (result > 1){
					cout << "Prob higher than 1:		" << result << endl;
				}
				if(result>maxProb){
					maxProb=result;
					maxProbRegion=adjacentReg;
				}
			}
		}
		// define merging threshold
		if(maxProb<0.5){
			maxProbRegion=-1;
		}
		return maxProbRegion;


	}
	//compute prior probabilites 
	vector<float> PriorProbs(){
		float SUMmerged=0;
		float PriorMerged, PriorMergedNot;					  //Priors;
		vector<float> output(2);		//1 PriorMerged, 2 PriorNotmerged

		for(int i=0; i < trainingData.size(); i++) {
			if(trainingData[i].isMerged){
				SUMmerged += 1;
			}
		}

		PriorMerged = SUMmerged / trainingData.size();
		PriorMergedNot = 1 - PriorMerged;
		output[0] = PriorMerged;
		output[1] = PriorMergedNot;


		return output;

	}
	//compute likelihood for a feature vector
	vector<float> likelihoods(float input, const int col, const float hmerged, const float hNotmerged){

		float likePnotMerged=0, likePmerged=0; 					//likelihoods
		vector<float>  trainDataMerged, trainDataNOTMerged;		//to copy data from trainingData
		double pi = 3.1415926535897;							//define pi for Gaussian KDE
		vector<float> output(2);								//vector to return likelihoods

		
		if(col==0){
			trainDataMerged=colorTDM;
			trainDataNOTMerged=colorTDNM;
		}
		else if(col==1){
			trainDataMerged=textureTDM;
			trainDataNOTMerged=textureTDNM;
		}
		else{
			trainDataMerged=arrangementTDM;
			trainDataNOTMerged=arrangementTDNM;
		}

#pragma omp parallel sections
{		
	#pragma omp section
	{	
		for (int i=0; i < trainDataMerged.size(); i++){
			likePmerged += (1/(sqrt((2*pi*pow(hmerged,2)))))*exp(-((pow(abs(input - trainDataMerged[i]),2))/(2*hmerged*hmerged)));
		}
		likePmerged /= trainDataMerged.size();
	}
	#pragma omp section
	{
		//calculate probability density of likelihood P(x|mergeNOT)
		for (int i=0; i < trainDataNOTMerged.size(); i++){
			likePnotMerged += (1/(sqrt((2*pi*pow(hNotmerged,2)))))*exp(-((pow(abs(input - trainDataNOTMerged[i]),2))/(2*hNotmerged*hNotmerged)));
		}
		likePnotMerged /= trainDataNOTMerged.size();
	}
}		
		//hand over values of the likelihood values to output vector
		output[0] = likePmerged;
		output[1] = likePnotMerged;

		return output;

	}

	void calculateLikelihoodDataVectors(){

		//For Color
		for(int i=0; i < trainingData.size(); i++){
			if(trainingData[i].isMerged){
				colorTDM.push_back( trainingData[i].values[0]);
				
			}
			else{
				colorTDNM.push_back(trainingData[i].values[0]);
				
			}
		}

		//For Texture
		for(int i=0; i < trainingData.size(); i++){
			if(trainingData[i].isMerged){
				textureTDM.push_back( trainingData[i].values[1]);
				
			}
			else{
				textureTDNM.push_back(trainingData[i].values[1]);
			
			}
		}

		//For Arrangement
		for(int i=0; i < trainingData.size(); i++){
			if(trainingData[i].isMerged){
				arrangementTDM.push_back( trainingData[i].values[2]);
				
			}
			else{
				arrangementTDNM.push_back(trainingData[i].values[2]);
				
			}
		}
	}

	

	void RunSLIC(const char* imagePath,int noOfSuperpixels,int nc){
		cout<<"\nRunning SLIC...\t\t\t\t";

		image=imread(imagePath,CV_LOAD_IMAGE_UNCHANGED);
		cvtColor( image, gray_image, COLOR_BGR2GRAY );
		img = cvLoadImage(imagePath, 1);
		IplImage *lab_image = cvCloneImage(img);
		cvCvtColor(img, lab_image, CV_BGR2Lab);


		/* Yield the number of superpixels and weight-factors from the user. */
		int w = img->width, h = img->height;
		double step = sqrt((w * h) / (double) noOfSuperpixels);

		/* Perform the SLIC superpixel algorithm. */
		Slic slic;
		slic.generate_superpixels(lab_image, step, nc);
		slic.create_connectivity(lab_image);

		for(int i=0;i<noOfSuperpixels+1000;i++){
			//Region newRegion(&image);
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

	void Perform(){
		cout<<"\nPerforming...\t\t\t\t";


		for(int a=1;noOfRegions>1;a++){

			bool flag=false;

			for(int i=0;i<adjacentRegions.size();i++){
				if(adjacentRegions[i].size()!=0){

					int mergeTo=Probability(i);
					if(mergeTo!=-1){


						int actualRegion=regionsCounter[i];
						Merge(actualRegion,mergeTo);

						flag=true;
					}			
				}
			}

			if(!flag)
				break;
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


	void SaveBoundaryMapImage(const char *boundaryMapImagePath){

		IplImage* boundaryMap = cvCreateImage(cvSize(image.size().width,image.size().height), 8, 1);

		cvZero(boundaryMap);

		int count=0;
		for (int i = 0; i < regions.size(); i++) {
			if(regionsCounter[i]==i){
				count++;
				for (int j = 0; j < regions[i].getNoOfBoundaryPixels(); j++) {
					Coord pixel=regions[i].getBoundaryPixel(j);
					int row=pixel.row, col=pixel.col;
					cvSet2D(boundaryMap, row,col, 255);
				}
			}
		}		

		cvSaveImage(boundaryMapImagePath,boundaryMap);


	}

	void ClearData(){
		regions.clear();
		regionsLabels.clear();
		regionsCounter.clear();
		adjacentRegions.clear();
		trainingData.clear();		
	}

public:
	RegionGrowing(){
		checkMatrix = Mat::zeros(image.size(), CV_8UC1);
		
	}	


	void Start(string name,const char* imagePath,int noOfSuperpixels,int nc,string adjacentRegionsFile,bool isAdjacentFile,string trainingFile,const char* resultImagePath,const char *boundaryMapImagePath){

		cout<<"\t\t\t  * Testing "<<name<<" *\n";

		Timer totalTime,localTime;
		totalTime.Start();

		localTime.Start();
		RunSLIC(imagePath,noOfSuperpixels,nc);	
		localTime.LocalEnd();
		cout<<"   Superpixels: "<<regions.size();
		
		localTime.Start();		
		if(isAdjacentFile)
			ReadAdjacentRegionsFromFile(adjacentRegionsFile);
		else
			FindAdjacentRegions(adjacentRegionsFile);
		localTime.LocalEnd();



		Training train;
		trainingData=train.GetTrainingData(trainingFile);
		cout<<"\nTraining Size: "<<trainingData.size();
		Priors=PriorProbs();
		//PriorProbs(Priors);
		calculateLikelihoodDataVectors();

		localTime.Start();
		Perform();
		localTime.LocalEnd();

		SaveResult(resultImagePath);



		SaveBoundaryMapImage(boundaryMapImagePath);

		ClearData();

		totalTime.TotalEnd();
	}

};

#endif