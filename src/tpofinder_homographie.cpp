/**
* Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
*
* MIT License (http://www.opensource.org/licenses/mit-license.php)
*/

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

//#include <hector_tpofinder/configure.h>
#include <tpofinder/detect.h>
#include <tpofinder/provide.h>
#include <tpofinder/visualize.h>

#include "tpofinder/util.h"
#include "tpofinder/truth.h"

#include <ros/ros.h>

using namespace cv;
using namespace tpofinder;
using namespace std;
//namespace po = boost::program_options;

const string NAME = "tpofind";

bool verbose = false;
bool webcam = false;
vector<string> files;

int main(int argc, char* argv[]) {

  ros::init(argc, argv, "tpofinder_homography");

  cvStartWindowThread();
  namedWindow("tpofinder_homography", WINDOW_NORMAL);

  ros::NodeHandle nh("~");

  string model_path;
  nh.getParam("model_path", model_path);


  std::vector<string> p_files;
  nh.getParam("files", p_files);

  ROS_INFO("File size: %d", p_files.size());

  for(unsigned i=0; i < p_files.size(); i++) {
    ROS_INFO("Including test file %s", p_files[i].c_str());
    files.push_back(p_files[i]);
  }

  HomographySequenceEstimator estimator;
  Mat firstImage;
  for (int i = 0; i < files.size(); i++) {

    ROS_INFO("Processing file %s", files[i].c_str());
    bool isFirst = false;
    Mat image = imread(files[i], 0);
    if (firstImage.empty()) {
      firstImage = image;
      isFirst = true;
    }
    Mat homography = estimator.next(image);

    Mat out = blend(firstImage, image, homography);

    if(!isFirst){
      stringstream ss;
      ss << setw(3) << setfill('0') << i;
      string file = ss.str();
      string homography_filename = model_path+"/"+file+".yml";
      writeHomography(homography_filename, homography);
      ROS_INFO("homography writen to %s", homography_filename.c_str());

      string newimg_filename = model_path+"/"+file+".jpg";
      imwrite(model_path+"/"+file+".jpg", imread(files[i], 0));
      ROS_INFO("file saved to %s", newimg_filename.c_str());
    }
    imshow("tpofinder_homography", out);
    waitKey(0);
  }

  return 0;
}
