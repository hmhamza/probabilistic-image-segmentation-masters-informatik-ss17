
//These are some of the structs that we used at different places in our code


#ifndef ONEREGION_H
#define ONEREGION_H

struct OneRegion{
	int region;
	float arrangement;

	OneRegion(int r,float a){
		region=r;
		arrangement=a;
	}
};

#endif


#ifndef TWOREGIONS_H
#define TWOREGIONS_H

struct TwoRegions{
	int region1;
	int region2;
	float arrangement;

	TwoRegions(int r1,int r2,float a){
		region1=r1;
		region2=r2;
		arrangement=a;
	}
};

#endif


#ifndef STAT_H
#define STAT_H

struct Stat{
	float values[3];	// 0)colorDiff	1) textureDiff	2) arrangement
	bool isMerged;

	Stat(float cd,float td,float a,bool im){
		values[0]=cd;
		values[1]=td;
		values[2]=a;
		isMerged=im;
	}
};

#endif
