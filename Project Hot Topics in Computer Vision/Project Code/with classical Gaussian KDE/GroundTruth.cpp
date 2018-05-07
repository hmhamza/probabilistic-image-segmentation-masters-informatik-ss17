//#include "stdafx.h"

/* internal includes */
#include "GroundTruth.h"
#include "Region.h"

/* standard library includes */
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

/* opencv includes */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

GroundTruth::GroundTruth()
{
}

GroundTruth::GroundTruth(int rows, int cols)
{
	std::vector<std::vector<int>> nCD(rows, std::vector<int>(cols, -1));
	this->nonCompressedData = nCD;
}

GroundTruth::GroundTruth(std::vector<Region> *regions, int rows, int cols)
{
	
	std::vector<std::vector<int>> nCD(rows, std::vector<int>(cols, -1));
	this->nonCompressedData = nCD;
	
	for(size_t regionIdx = 0; regionIdx < regions->size(); regionIdx++)
	{
		for(size_t coordIdx = 0; coordIdx < regions->at(regionIdx).getNoOfPixels(); coordIdx++)
		{
			Coord coord = regions->at(regionIdx).getAt(coordIdx);
			this->nonCompressedData.at(coord.row).at(coord.col) = regionIdx;
		}
	}
	
	numberOfSegments = regions->size();
}


int
GroundTruth::getSegmentOfPixel(int row, int col) {
	return this->nonCompressedData.at(row).at(col);
}

void GroundTruth::setSegmentOfPixel(int value, int row, int col)
{
	this->nonCompressedData.at(row).at(col) = value;
}

void GroundTruth::setNumberOfSegments(int number)
{
	this->numberOfSegments = number;
}

void
GroundTruth::importData(char* fileName) {

	std::ifstream file;
	file.open(fileName);

	bool data = false;
	std::string line;

	size_t width = 0;
	size_t height = 0;

	if (file.is_open())
	{
		while (std::getline(file, line))
		{
			if (!data)
			{
				int posFirstSpace = line.find(' ');
				std::string metaData = line.substr(0, posFirstSpace);
				std::string data = line.substr(posFirstSpace + 1);
				if (metaData.compare("width") == 0)
					width = std::stoi(data);
				if (metaData.compare("height") == 0)
					height = std::stoi(data);
				if (metaData.compare("segments") == 0)
					this->numberOfSegments = std::stoi(data);
			}
			if (data)
			{
				/* get data */
				std::stringstream lineSS;
				lineSS.str(line);
				std::string segment;
				std::vector<int> data;
				while (std::getline(lineSS, segment, ' '))
					data.push_back(std::stoi(segment));
				/* write data */
				int row = data.at(1);
				int seg = data.at(0);
				for (int column = data.at(2); column <= data.at(3); column++)
					this->nonCompressedData.at(row).at(column) = seg;
			}
			if (line.compare("data") == 0)
			{
				data = true;
				/* initialize data structure */
				std::vector< std::vector<int> > nCD(height, std::vector<int>(width));
				this->nonCompressedData = nCD;
			}
		}
		file.close();
	}
	else
		std::cout << "failed to open file" << std::endl;
}

void 
GroundTruth::showGroundTruth()
{
	size_t height = this->nonCompressedData.size();
	size_t width = this->nonCompressedData.at(0).size();
	cv::Mat A(height, width, CV_64F);
	for (size_t height_ = 0; height_ < height; height_++)
		for (size_t width_ = 0; width_ < width; width_++)
			A.at<double>(height_, width_) = this->nonCompressedData.at(height_).at(width_)*(1.0 / this->numberOfSegments);
	cv::imshow("", A);
	cv::waitKey(0);
}

void 
GroundTruth::printGroundTruth()
{
	for (size_t height = 0; height < this->nonCompressedData.size(); height++)
	{
		for (size_t width = 0; width < this->nonCompressedData.at(0).size(); width++)
			std::cout << this->nonCompressedData.at(height).at(width) << "\t";
		std::cout << std::endl;
	}
}
