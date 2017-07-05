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

#include <ros/ros.h>

using namespace cv;
using namespace tpofinder;
using namespace std;
//namespace po = boost::program_options;

const string NAME = "tpofind";

bool verbose = false;
bool webcam = false;
vector<string> files;

struct RosDebugFilter : public DetectionFilter {

  RosDebugFilter(const cv::Ptr<DetectionFilter> filter)
  /*    */ : filter_(filter){
}

virtual bool accept(const Detection & d){
  ROS_INFO("Detection details:");
  ROS_INFO("Number of Inliers: %d", d.inliers.size());
  ROS_INFO("Number of Matches: %d", d.matches.size());
  ROS_INFO("Ratio: %f", (d.inliers.size() / (double) d.matches.size()));
  return filter_->accept(d);
}

private:
  cv::Ptr<DetectionFilter> filter_;

};

void loadModel(Modelbase& modelbase, const string& path) {
  if (verbose) {
    cout << boost::format("Loading object %-20s ... ") % path;
  }
  modelbase.add(path);
  if (verbose) {
    cout << "[DONE]" << endl;
  }
}

void processImage(Detector& detector, Mat &image) {
  if (!image.empty()) {
    cout << "Detecting objects on image          ... ";
    Scene scene = detector.describe(image);
    vector<Detection> detections = detector.detect(scene);
    cout << "[DONE]" << endl;
    cout << "Found " << detections.size() << " objects" << endl;

    BOOST_FOREACH(Detection d, detections) {
      drawDetection(image, d);
    }
  }
}

int main(int argc, char* argv[]) {

  ros::init(argc, argv, "tpofinder_test");

  cvStartWindowThread();
  namedWindow(NAME, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);

  // TODO: remove duplication
  // TODO: support SIFT
  // TODO: make customizable
  Ptr<FeatureDetector> fd = ORB::create(1000);
  Ptr<FeatureDetector> trainFd = ORB::create(250);
  Ptr<DescriptorExtractor> de = ORB::create(1000);

  Ptr<flann::IndexParams> indexParams = new flann::LshIndexParams(15, 12, 2);
  Ptr<DescriptorMatcher> dm = new FlannBasedMatcher(indexParams);

  Feature trainFeature(trainFd, de, dm);

  Modelbase modelbase(trainFeature);

  ros::NodeHandle nh("~");

  std::vector<string> models;
  nh.getParam("models", models);
  for(unsigned i=0; i < models.size(); i++) {
    ROS_INFO("Loading model %s", models[i].c_str());
    loadModel(modelbase, models[i]);
  }

  Feature feature(fd, de, dm);
  /*
  Ptr<DetectionFilter> filter = new AndFilter(
  Ptr<DetectionFilter> (new EigenvalueFilter(-1, 4.0)),
  Ptr<DetectionFilter> (new InliersRatioFilter(0.30)));
  */
  Ptr<DetectionFilter> filter = new RosDebugFilter(new InliersRatioFilter(0.1));

  Detector detector(modelbase, feature, filter);

  std::vector<string> p_files;
  nh.getParam("files", p_files);

  ROS_INFO("File size: %d", p_files.size());

  for(unsigned i=0; i < p_files.size(); i++) {
    ROS_INFO("Including test file %s", p_files[i].c_str());
    files.push_back(p_files[i]);
  }

  ImageProvider *image_provider = new ListFilenameImageProvider(files);

  Mat image;
  while (image_provider->next(image)) {
    clock_t start = clock();
    processImage(detector, image);
    clock_t ticks = clock()-start;
    cout << "Detection time: " << (double)ticks/CLOCKS_PER_SEC << endl;

    if (!image.empty()) {
      imshow(NAME, image);
    }
    waitKey(0);
  }

  delete image_provider;

  if (verbose) {
    cout << "No more images to process           ... [DONE]" << endl;
  }

  cout << "Waiting for key (win) or CTRL+C     ... [DONE]" << endl;
  while (waitKey(10) == -1) {
    imshow(NAME, image);
  }

  if (verbose) {
    cout << "Quitting                            ... [DONE]" << endl;
  }

  return 0;
}
