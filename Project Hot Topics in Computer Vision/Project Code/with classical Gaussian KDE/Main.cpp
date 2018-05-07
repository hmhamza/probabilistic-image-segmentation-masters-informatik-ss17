#include "RegionGrowing.h"
#include "Training.h"
#include"Timer.h"

/* This function to do training with multiple files */
void doTraining(){

	vector<string> trainingFiles;
	trainingFiles.push_back("368016");
	trainingFiles.push_back("365025");
	trainingFiles.push_back("2092");
	trainingFiles.push_back("8143");
	trainingFiles.push_back("12074");
	trainingFiles.push_back("368078");
	trainingFiles.push_back("374020");
	trainingFiles.push_back("388016");
	trainingFiles.push_back("239007");
	trainingFiles.push_back("242078");

	Training train;
	for(int i=0;i<trainingFiles.size();i++){
		string imgPath="Training/New/"+trainingFiles[i]+".jpg";
		string resultImgPath="Training/New/"+trainingFiles[i]+"_Result.jpg";
		
		//The boolean value (6th parameter) in this function takes false if this image is going to be used for the first time, else takes true. The reason is, if that image was processed before, then we would have saved the Adjacent regions data for it.
		//So now, we can just get the Adjacent regions data from a file and save computation time
		train.Start(trainingFiles[i],"Training/New/"+trainingFiles[i]+".seg",imgPath.c_str(),100,1,"Training/New/"+trainingFiles[i]+"_AdjacentRegions_100_1.txt",false,"Training/New/"+trainingFiles[i]+"_Statistics.txt",resultImgPath.c_str());		
	}	
}

/* This function is used to test multiple Images, one at a time in a loop */
void doTesting(){

	vector<string> testingFiles;

	//testingFiles.push_back("41033");

	/*
	testingFiles.push_back("145086");
	testingFiles.push_back("147091");
	testingFiles.push_back("148026");
	testingFiles.push_back("148089");
	testingFiles.push_back("156065");
	testingFiles.push_back("157055");
	testingFiles.push_back("159008");
	testingFiles.push_back("160068");
	testingFiles.push_back("163085");
	testingFiles.push_back("167062");
	testingFiles.push_back("167083");
	testingFiles.push_back("170057");
	testingFiles.push_back("175032");
	testingFiles.push_back("175043");
	testingFiles.push_back("182053");
	testingFiles.push_back("189080");
	testingFiles.push_back("196073");
	testingFiles.push_back("197017");
	testingFiles.push_back("208001");
	testingFiles.push_back("210088");
	testingFiles.push_back("216081");
	testingFiles.push_back("219090");
	testingFiles.push_back("220075");
	testingFiles.push_back("223061");
	testingFiles.push_back("227092");
	testingFiles.push_back("229036");
	testingFiles.push_back("236037");
	testingFiles.push_back("241004");
	testingFiles.push_back("241048");
	testingFiles.push_back("253027");
	testingFiles.push_back("253055");
	testingFiles.push_back("260058");
	testingFiles.push_back("271035");
	testingFiles.push_back("285079");
	testingFiles.push_back("291000");
	
	//testingFiles.push_back("295087");
	//testingFiles.push_back("296007");
	testingFiles.push_back("296059");
	testingFiles.push_back("299086");
	testingFiles.push_back("300091");
	testingFiles.push_back("302008");
	testingFiles.push_back("304034");
	testingFiles.push_back("304074");
	*/
	testingFiles.push_back("241004");
	testingFiles.push_back("3063");
	testingFiles.push_back("8068");
	testingFiles.push_back("145059");
	
	RegionGrowing obj;
	for(int i=0;i<testingFiles.size();i++){
		string testImg="Testing/Images/"+testingFiles[i]+".jpg";
		string resultImg="Testing/Results/"+testingFiles[i]+".jpg";
		string boundaryMapImg="Testing/Boundary Maps/"+testingFiles[i]+"_BoundaryMap.png";
		
		//The boolean value (6th parameter) in this function takes false if this image is going to be used for the first time, else takes true. The reason is, if that image was processed before, then we would have saved the Adjacent regions data for it.
		//So now, we can just get the Adjacent regions data from a file and save computation time
		obj.Start(testingFiles[i],testImg.c_str(),1500,1,"Testing/Adjacent Files/"+testingFiles[i]+"_AdjacentRegions_1500_1.txt",false,"Training/New/Final.txt",resultImg.c_str(),boundaryMapImg.c_str());
	}		
}

int main(){
	
	//doTraining();

	doTesting();

	cout << "finished" << endl;
	return 0;
}
