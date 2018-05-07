#pragma once

/* standard library includes */
#include <vector>
#include "Region.h"

class GroundTruth
{
public:
    GroundTruth();
    GroundTruth(int rows, int cols);
    GroundTruth(std::vector<Region> *Regions, int rows, int cols);
    //~GroundTruth();
    
    int getSegmentOfPixel(int row, int col);
    void setSegmentOfPixel(int value, int row, int col);
    void setNumberOfSegments(int number);
    
    void importData(char* fileName);
    void showGroundTruth();
    void printGroundTruth();
    
private:
    std::vector< std::vector<int> > nonCompressedData; /* full sized image, pixel represent the segment */
    int numberOfSegments;
};

