#ifndef REGIONGROWING_H
#define REGIONGROWING_H

#include <iostream>
#include<vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include<iostream>
#include <algorithm>

#include"Training.h"

using namespace std;
using namespace cv;

class RegionGrowing{

	IplImage *img;

	Mat image,gray_image;
	vector <Region> regions;
	vector<vector<int>> regionsLabels;

	vector<int> regionsCounter;						//To keep data which regions have been merged
	vector<vector<OneRegion>> adjacentRegions;
	vector<Stat> trainingData;						//Training data which is got from Training.h
	vector<float> Priors;
	int noOfRegions;

	vector<float> colorTDM,colorTDNM,textureTDM,textureTDNM,arrangementTDM,arrangementTDNM;		//TDM: Training Data Merged		TDNM: Training Data Not Merged
	//define Bterms for all features
	vector<float> Bterms_colorTDM, Bterms_colorTDNM,Bterms_textureTDM,Bterms_textureTDNM,Bterms_arrangementTDM,Bterms_arrangementTDNM;

	//define Cluster indices for all features
	vector<int> 	pClusterIndex_colorTDM, 	pClusterIndex_colorTDNM;
	vector<int> 	pClusterIndex_textureTDM,pClusterIndex_textureTDNM;
	vector<int> 	pClusterIndex_arrangementTDM,pClusterIndex_arrangementTDNM;
	//define cluster centers for all features
	vector<float> pClusterCenter_colorTDM, pClusterCenter_colorTDNM;
	vector<float> pClusterCenter_textureTDM, pClusterCenter_textureTDNM;
	vector<float> pClusterCenter_arrangementTDM, pClusterCenter_arrangementTDNM;
	//define A-terms for all features
	vector<float> a_terms_colorTDM, a_terms_colorTDNM;
	vector<float> a_terms__textureTDM, a_terms_textureTDNM;
	vector<float> a_terms_arrangementTDM, a_terms_arrangementTDNM;

	//define parameters for all features
	float rx_colorTDM, rx_colorTDNM, rx_textureTDM, rx_textureTDNM, rx_arrangementTDM, rx_arrangementTDNM;
	float K_colorTDM, K_colorTDNM, K_textureTDM, K_textureTDNM, K_arrangementTDM, K_arrangementTDNM;
	float p_colorTDM, p_colorTDNM, p_textureTDM, p_textureTDNM, p_arrangementTDM, p_arrangementTDNM;
	float q_colorTDM, q_colorTDNM, q_textureTDM, q_textureTDNM, q_arrangementTDM, q_arrangementTDNM;

	//define bandwidth of the KDE for all features (use provided matlab program to calculate bandwith)
	double hColor = 0.0118, hColorNOT = 0.0221; //0.0083, NOT 0.0174
	double hText =  0.0342, hTextNOT = 0.0310; // 0.0346, 0.0270
	double hArr=  0.0113, hArrNOT = 0.0053;		//0.0068 6.2113e-04
	//define derivative of Gaussian KDE - NB:LEAVE r=0
	int r=0;
	//define desired error
	float eps= 4.4e-6;
	//define max and min values for scaling
	double maxColor, minColor;
	double maxTexture, minTexture;
	double maxArrangement, minArrangement;


	Mat checkMatrix;

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

	/* Display contours of all regions by chaging the color of the boundary pixels to red */
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

	/* Display contours of a specific region by chaging the color of the boundary pixels of that region to red */
	void displayBoundaryOfaRegion(int i,char *name){
		IplImage *contourImage = cvCloneImage(img);
		for (int j = 0; j < regions[i].getNoOfBoundaryPixels(); j++) {
			Coord pixel=regions[i].getBoundaryPixel(j);
			int row=pixel.row, col=pixel.col;
			cvSet2D(contourImage, row,col, CV_RGB(255,0,0));	
		}

		
		cvShowImage(name,contourImage);
		

	}

	/* Merges 2 regions whose labels are passed as params */
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

	
	//calculation of posterior probability
	int Probability(int reg) {
		float maxProb =-1;
		int actualRegion=regionsCounter[reg];


		Region *current=&regions[actualRegion];


		int maxProbRegion=0;
		for(int i=0;i<adjacentRegions[reg].size();i++){
			int adjacentReg=adjacentRegions[reg][i].region;
			adjacentReg=regionsCounter[adjacentReg];
			if(actualRegion!=adjacentReg){


				Region *adjacent= &regions[adjacentReg];
				float Diffcolor=sqrt(pow((current->calculateGLColorMeanValue(0)- adjacent->calculateGLColorMeanValue(0)),2)+
					pow((current->calculateGLColorMeanValue(1)- adjacent->calculateGLColorMeanValue(1)),2)+
					pow((current->calculateGLColorMeanValue(2)- adjacent->calculateGLColorMeanValue(2)),2));
				Diffcolor = (Diffcolor -minColor)/(maxColor-minColor);

				float DiffTexture=abs(current->calculateGLTexture()-adjacent->calculateGLTexture());
				DiffTexture = (DiffTexture -minTexture)/(maxTexture-minTexture);

				float Arrangement=adjacentRegions[reg][i].arrangement;
				Arrangement = (Arrangement-minArrangement)/(maxArrangement-minArrangement);

				float result=0; 
				float LikeColor, LikeTexture, LikeArrangement;
				float LikeColorNOT, LikeTextureNOT, LikeArrangementNOT;




				LikeColor = likelihoods(p_colorTDM, r, Diffcolor, rx_colorTDM, K_colorTDM, hColor, pClusterCenter_colorTDM, Bterms_colorTDM, a_terms_colorTDM);
				LikeColorNOT = likelihoods(p_colorTDNM, r, Diffcolor, rx_colorTDNM, K_colorTDNM, hColorNOT, pClusterCenter_colorTDNM, Bterms_colorTDNM, a_terms_colorTDNM);


				LikeTexture = likelihoods(p_textureTDM, r, DiffTexture, rx_textureTDM, K_textureTDM, hText, pClusterCenter_textureTDM, Bterms_textureTDM, a_terms__textureTDM);
				LikeTextureNOT = likelihoods(p_textureTDNM, r, DiffTexture, rx_textureTDNM, K_textureTDNM, hTextNOT, pClusterCenter_textureTDNM, Bterms_textureTDNM, a_terms_textureTDNM);

				LikeArrangement = likelihoods(p_arrangementTDM, r, Arrangement, rx_arrangementTDM, K_arrangementTDM, hArr, pClusterCenter_arrangementTDM, Bterms_arrangementTDM, a_terms_arrangementTDM);
				LikeArrangementNOT = likelihoods(p_arrangementTDNM, r, Arrangement, rx_arrangementTDNM, K_arrangementTDNM, hArrNOT, pClusterCenter_arrangementTDNM, Bterms_arrangementTDNM, a_terms_arrangementTDNM);



				result = (LikeColor*LikeTexture*LikeArrangement*Priors[0])/
					((LikeColor*LikeTexture*LikeArrangement*Priors[0])+(LikeColorNOT*LikeTextureNOT*LikeArrangementNOT*Priors[1]));

				


				if (result > 1){
					cout << "Prob higher than 1:		" << result << endl;
				}
				if(result>maxProb){
					maxProb=result;
					maxProbRegion=adjacentReg;
				}
			}
		}
		//definition of merging probability
		if(maxProb < 0.5){
			maxProbRegion=-1;
		}
		return maxProbRegion;


	}


	//calculation of prior probabilties 
	vector<float> PriorProbs(){

		float SUMmerged=0;
		float PriorMerged, PriorMergedNot;					  //Priors;
		vector<float> output(2);		//1 PriorMerged, 2 PriorNotmerged

		//calculate number of merge events
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
	


	//calculate likelihood of a feature
	float likelihoods(int p, int r, float input, double rx, int K,double h,vector<float> pClusterCenter,vector<float> B_terms, vector<float> a_terms)
	{

		
		float output=0;
		double *temp3;
		temp3=new double[p+r];

		float h_square = h*h;
		float two_h_square = 2*h_square;
		double r_term=sqrt((double)factorial(r));
		float rr = 2*h*sqrt(log(r_term/eps));
		float ry = rx+rr;


			int target_cluster_number=min((int)floor(input/rx),K-1);
	
			double temp1=input-pClusterCenter[target_cluster_number];
			double dist=abs(temp1);
			while (dist <= ry && target_cluster_number <K && target_cluster_number >=0){


				double temp2=exp(-temp1*temp1/two_h_square);
				double temp1h=temp1/h;
				temp3[0]=1;
				for(int i=1; i<p+r; i++){
					temp3[i]=temp3[i-1]*temp1h;
				}

				for(int k=0; k<=p-1; k++){
					int dummy=0;
					for(int l=0; l <= (int)floor((double)r/2); l++){
						for(int m=0; m <= r-(2*l); m++){
					
							output += (a_terms[dummy]*B_terms[(target_cluster_number*p*(r+1))+((r+1)*k)+m]*temp2*temp3[k+r-(2*l)-m]);
							dummy +=1;
						}
					}
				}
				


				target_cluster_number +=1;;
				temp1=input-pClusterCenter[target_cluster_number];
				dist=abs(temp1);
			}

			target_cluster_number=min((int)floor(input/rx),K-1)-1;
			if (target_cluster_number >=0){
				double temp1= input -pClusterCenter[target_cluster_number];
				double dist=abs(temp1);
				while (dist <= ry && target_cluster_number <K && target_cluster_number >=0){
			
					double temp2=exp(-(temp1*temp1/two_h_square));
				    double temp1h=temp1/h;
					temp3[0]=1;
					for(int i=1; i<p+r; i++){
						temp3[i]=temp3[i-1]*temp1h;
					}

					for(int k=0; k<=p-1; k++){
						int dummy=0;
						for(int l=0; l <= (int)floor((double)r/2); l++){
							for(int m=0; m <= r-(2*l); m++){
								output += (a_terms[dummy]*B_terms[(target_cluster_number*p*(r+1))+((r+1)*k)+m]*temp2*temp3[k+r-(2*l)-m]);
								dummy++;
							}
						}
					}
					
					target_cluster_number -=1;
					temp1= input -pClusterCenter[target_cluster_number];
					dist=abs(temp1);
				}
			}


		

		return output;
	}


	//function to compute facotrial
	double factorial(int n){

		int fact=1;
		if (n == 0)
			return 1;
		else{
			for ( int i = 1; i <= n; i++){
				fact=fact*i;
			}
		}
		return fact;
	}


	//computation of the parameters of the Fast Gaussian KDE approximation
	vector<float>  choose_parameters(double h, int N){

		vector<float> output(4);
		//rx: interval length.

		double rx=h/2;

		//K: number of intervals.

		int K=(int)ceil(1.0/rx);
		rx= 1.0/K;
		double rx_square= rx*rx;
		double h_square = h*h;
		double two_h_square = 2*h_square;

		//rr: cutoff radius.
		double r_term=sqrt((double)factorial(r));


		double rr = 2*h*sqrt(log(r_term/eps));

		//ry: cluster cutoff radius.
		double ry = rx+rr;

		//p: truncation number.

		int p=0;
		double error=1;
		double temp=1;
		double comp_eps=eps/r_term;

		while((error > comp_eps) & (p <= 500)){
			p+=1;
			double b=min(((rx+sqrt((rx_square)+(8*p*h_square)))/2),ry);
			double c=rx-b;
			temp=temp*(((rx*b)/h_square)/p);
			error=temp*(exp(-(c*c)/2*two_h_square));
		}
		p=p+1;
		//calculate q
		double pi=3.14159265358979;
		double q=(pow(-1,r))/(sqrt(2*pi)*N*(pow(h,(r+1))));
		output[0] = rx;
		output[1] = K;
		output[2] = p;
		output[3] = q;


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

	//compute max and min values of feature vector. Required to normalize the training data 
	void CalcVectorMaxandMin (int col, vector<Stat> featureVector){
		float max=1;
		for(int i=0; i < featureVector.size();i++){
				if (featureVector[i].values[col] > max){
					max = featureVector[i].values[col];
				}
			}
		float min = max;
		for(int i=0; i < featureVector.size();i++){
				if (featureVector[i].values[col] < min){
					min = featureVector[i].values[col];
				}
			}
		switch ( col )  {
			case 0:
			{
				maxColor = max;
				minColor = min;
				break;
			}
			case 1:
			{
				maxTexture = max;
				minTexture = min;
				break;
			}
			case 2:
			{
				maxArrangement = max;
				minArrangement = min;
				break;
			}

		}




	}

	void  normalizeTrainVector(vector<Stat>& input){
	int numberfeatures =3;
	for (int col=0; col< numberfeatures; col++){
		float max=0;
				for(int i=0; i < input.size();i++){
					if (input[i].values[col] > max){
						max = input[i].values[col];
					}
				}
				float min = max;
				for(int i=0; i < input.size();i++){
					if (input[i].values[col] < min){
								min = input[i].values[col];
							}
						}
				for (int i=0; i < input.size();i++){
					input[i].values[col] = (input[i].values[col] -min)/(max-min);
				}
		}
				
	}
	
	//define Gaussian KDE approximation parameters for each feature vector
	void calculateParameters(){

		//parameters for colorTDM
		vector<float> ColorTDM_parameters = choose_parameters( hColor, colorTDM.size());
		rx_colorTDM = ColorTDM_parameters[0];
		K_colorTDM  = ColorTDM_parameters[1];
		p_colorTDM  = ColorTDM_parameters[2];
	    q_colorTDM  = ColorTDM_parameters[3];

	    //parameters for colorTDNM
	    vector<float> ColorTDNM_parameters = choose_parameters( hColorNOT, colorTDNM.size());
	    rx_colorTDNM = ColorTDNM_parameters[0];
		K_colorTDNM  = ColorTDNM_parameters[1];
		p_colorTDNM  = ColorTDNM_parameters[2];
		q_colorTDNM	=  ColorTDNM_parameters[3];

		//parameters for textureTDM
		vector<float> TextureTDM_parameters = choose_parameters( hText, textureTDM.size());
		rx_textureTDM =TextureTDM_parameters[0];
		K_textureTDM = TextureTDM_parameters[1];
		p_textureTDM = TextureTDM_parameters[2];
		q_textureTDM = TextureTDM_parameters[3];

		//parameters for textureTDNM
		vector<float> TextureTDNM_parameters = choose_parameters( hTextNOT, textureTDNM.size());
		rx_textureTDNM = TextureTDNM_parameters[0];
		K_textureTDNM = TextureTDNM_parameters[1];
		p_textureTDNM = TextureTDNM_parameters[2];
		q_textureTDNM = TextureTDNM_parameters[3];

		//parameters for ArrangementTDM
		vector<float> ArrangementTDM_parameters = choose_parameters( hArr, arrangementTDM.size());
		rx_arrangementTDM = ArrangementTDM_parameters[0];
		K_arrangementTDM = ArrangementTDM_parameters[1];
		p_arrangementTDM = ArrangementTDM_parameters[2];
		q_arrangementTDM = ArrangementTDM_parameters[3];

		//parameters for ArrangementTDNM
		vector<float> ArrangementTDNM_parameters = choose_parameters( hArrNOT, arrangementTDNM.size());
		rx_arrangementTDNM = ArrangementTDNM_parameters[0];
		K_arrangementTDNM = ArrangementTDNM_parameters[1];
		p_arrangementTDNM = ArrangementTDNM_parameters[2];
		q_arrangementTDNM = ArrangementTDNM_parameters[3];

	}

	//compute the cluster centers of intervals (needed for Fast Gaussian KDE approximation)
	void calculateLikelihoodCenters(){
		//calculate cluster center for color
		CalcClusterCenter(rx_colorTDM, K_colorTDM, pClusterCenter_colorTDM);
		CalcClusterCenter(rx_colorTDNM, K_colorTDNM, pClusterCenter_colorTDNM);

		//calculate cluster center for texture
		CalcClusterCenter(rx_textureTDM, K_textureTDM,pClusterCenter_textureTDM);
		CalcClusterCenter(rx_textureTDNM, K_textureTDNM, pClusterCenter_textureTDNM);

		//calculate cluster center for arrangement
		CalcClusterCenter(rx_arrangementTDM, K_arrangementTDM, pClusterCenter_arrangementTDM);
		CalcClusterCenter(rx_arrangementTDNM,K_arrangementTDNM, pClusterCenter_arrangementTDNM);

	}
	void CalcClusterCenter(float rx, float K, std::vector<float>& input){

		for(int i=0; i<K; i++){
			input.push_back((i*rx)+(rx/2));
	


		}

		
	}


	//calculate corrisponidng cluster indices for each entry of the training data (needed for Fast Gaussian KDE approximation)
	void calculateLikelihoodIndices(){
	
		CalcClusterIndex(rx_colorTDM, K_colorTDM, colorTDM, pClusterIndex_colorTDM );
		CalcClusterIndex(rx_colorTDNM, K_colorTDNM, colorTDNM, pClusterIndex_colorTDNM);

		CalcClusterIndex(rx_textureTDM, K_textureTDM, textureTDM, pClusterIndex_textureTDM);
		CalcClusterIndex(rx_textureTDNM, K_textureTDNM, textureTDNM, pClusterIndex_textureTDNM);

		CalcClusterIndex(rx_arrangementTDM, K_arrangementTDM,arrangementTDM, pClusterIndex_arrangementTDM);
		CalcClusterIndex(rx_arrangementTDNM, K_arrangementTDNM,arrangementTDNM, pClusterIndex_arrangementTDNM);
	}

	void CalcClusterIndex(double rx, int K, vector<float>& inputVector, vector<int>& pClusterIndex){
		int N = inputVector.size();

		for(int i=0; i<N; i++){
			pClusterIndex.push_back(min((int)floor(inputVector[i]/rx),K-1));

		

		}




	}

	//define B-terms for each feature (needed for Fast Gaussian KDE approximation)
	void calculateLikelihoodBterms(){


		compute_Bterms(K_colorTDM, p_colorTDM, r, hColor, colorTDM.size(), q_colorTDM, pClusterIndex_colorTDM, pClusterCenter_colorTDM, colorTDM, Bterms_colorTDM);
		compute_Bterms(K_colorTDNM, p_colorTDNM, r, hColorNOT, colorTDNM.size(), q_colorTDNM, pClusterIndex_colorTDNM, pClusterCenter_colorTDNM, colorTDNM, Bterms_colorTDNM);

		compute_Bterms(K_textureTDM, p_textureTDM, r, hText, textureTDM.size(), q_textureTDM, pClusterIndex_textureTDM, pClusterCenter_textureTDM, textureTDM, Bterms_textureTDM);
		compute_Bterms(K_textureTDNM, p_textureTDNM, r, hTextNOT, textureTDNM.size(), q_textureTDNM, pClusterIndex_textureTDNM, pClusterCenter_textureTDNM, textureTDNM, Bterms_textureTDNM);

		compute_Bterms(K_arrangementTDM, p_arrangementTDM, r, hArr, arrangementTDM.size(), q_arrangementTDM, pClusterIndex_arrangementTDM, pClusterCenter_arrangementTDM, arrangementTDM, Bterms_arrangementTDM);
		compute_Bterms(K_arrangementTDNM, p_arrangementTDNM, r, hArrNOT, arrangementTDNM.size(), q_arrangementTDNM, pClusterIndex_arrangementTDNM, pClusterCenter_arrangementTDNM, arrangementTDNM,Bterms_arrangementTDNM);

		}
	//computation of the B-terms  (needed for Fast Gaussian KDE approximation)
	void compute_Bterms(int K, int p, int r, double h, int N, double q,  vector<int>  pClusterIndex,  vector<float> pClusterCenter,  vector<float> inputVec, vector<float>& BtermVec){



			int num_of_B_terms=K*p*(r+1);
			vector<float> resultVec(num_of_B_terms);
			double *k_factorial;
			k_factorial=new double[p];

			k_factorial[0]=1;
			for(int i=1; i<p ;i++){
				k_factorial[i]= k_factorial[i-1]/i;

			}

			double *temp3;
			temp3=new double[p+r];

			for(int n=0; n<K; n++){
				for(int k=0; k<p; k++){
					for(int m=0; m< r+1; m++){
						resultVec[(n*p*(r+1))+((r+1)*k)+m]=0.0;

					}
				}

			}




			for(int i=0; i<N; i++){
				int cluster_number=pClusterIndex[i];
				double temp1=(inputVec[i]-pClusterCenter[cluster_number])/h;
				double temp2=exp(-temp1*temp1/2);
				temp3[0]=1;
				for(int k=1; k<p+r; k++){
					temp3[k]=temp3[k-1]*temp1;
				}

				for(int k=0; k<p; k++){
					for(int m=0; m< r+1; m++){
						resultVec[(cluster_number*p*(r+1))+((r+1)*k)+m] += (temp2*temp3[k+m]);
					}
				}
			}

			for(int n=0; n<K; n++){
				for(int k=0; k<p; k++){
					for(int m=0; m< r+1; m++){
						resultVec[(n*p*(r+1))+((r+1)*k)+m] *=(k_factorial[k]*q);

					}
				}

			}

			for(int i=0; i<resultVec.size();i++){
				BtermVec.push_back(resultVec[i]);
			}





		}

	//define a-terms for each feature vector
	void calculateLikelihood_a_terms(){


		compute_a(r, a_terms_colorTDM);
		compute_a(r, a_terms_colorTDNM);

		compute_a(r, a_terms__textureTDM);
		compute_a(r, a_terms_textureTDNM);

		compute_a(r,a_terms_arrangementTDM);
		compute_a(r, a_terms_arrangementTDNM);

	}
	//computation of the a-terms  (needed for Fast Gaussian KDE approximation, especially for computing derivatives of the Gaussian)
	void compute_a(int r, vector<float>& a_termsOut){

			double r_factorial=(double)factorial(r);

		    double *l_constant;
			l_constant=new double[((int)floor((double)r/2))+1];
			l_constant[0]=1;
			for(int l=1; l <= (int)floor((double)r/2); l++){
				l_constant[l]=l_constant[l-1]*(-1.0/(2*l));

			}

			double *m_constant;
			m_constant=new double[r+1];
			m_constant[0]=1;
			for(int m=1; m <= r; m++){
				m_constant[m]=m_constant[m-1]*(-1.0/m);

			}

			int num_of_a_terms=0;
			for(int l=0; l <= (int)floor((double)r/2); l++){
				for(int m=0; m <= r-(2*l); m++){
					num_of_a_terms +=1;
				}
			}

			double *a_terms;
			a_terms= new double[num_of_a_terms];
			int k=0;
			for(int l=0; l <= (int)floor((double)r/2); l++){
				for(int m=0; m <= r-(2*l); m++){
					a_terms[k]=(l_constant[l]*m_constant[m]*r_factorial)/((double)factorial(r-(2*l)-m));
					k++;
					}
				}

			for(int i=0; i<num_of_a_terms; i++){
				a_termsOut.push_back(a_terms[i]);
			}

		}



	//Do first coarse segmentation with SLIC
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
	
			Region newRegion(&image,&gray_image);
			regions.push_back(newRegion);
		}

		slic.getResults(img,regions,regionsLabels);
		noOfRegions=regions.size();

		for(int i=0;i<regions.size();i++){
			regionsCounter.push_back(i);
		}

	}

	/* Find adjacent regions for all of the regions */
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

	/* Reads adjacent regions from a file if the file has already been populated */
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

	/* Performs the algorithm. Calculates the merging probabilities and performs merging */
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

	/* Save the segementation results for later view */
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

	/* Save the boundary map of the segmented image to be used for evaluation */
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


	/* The main driver function of the class */
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
		//Initializing functions
		Priors=PriorProbs();						//calculation of priors
		//normalization of trainingsdata
		CalcVectorMaxandMin ( 0, trainingData);		
		CalcVectorMaxandMin ( 1, trainingData);
		CalcVectorMaxandMin ( 2, trainingData);
		normalizeTrainVector(trainingData);
		calculateLikelihoodDataVectors();
		//calculation of parameters for the Fast Gaussaian approximation
		calculateParameters();
		calculateLikelihoodCenters();
		calculateLikelihoodIndices();
		calculateLikelihood_a_terms();
		calculateLikelihoodBterms();


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
