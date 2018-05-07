#pragma once

/* standard library includes */
#include <vector>
#include "Region.h"

/* class ground truth stores for each pixel of a picture the region it belongs to */
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
    
    /* read ground truth file of berkeley library */
    void importData(char* fileName);
    
    /* visualize ground truth as a picture, each region gets its own grey shade */
    void showGroundTruth();

    /* prints gound truth as a matrix (exactly as it is stored) */
    void printGroundTruth();
    
private:
    std::vector< std::vector<int> > nonCompressedData; /* full sized image, pixel represents the segment */
    int numberOfSegments;
};

