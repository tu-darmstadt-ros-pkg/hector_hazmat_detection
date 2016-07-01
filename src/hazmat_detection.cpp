//=================================================================================================
// Copyright (c) 2016, Christian Rose, TU Darmstadt
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Flight Systems and Automatic Control group,
//       TU Darmstadt, nor the names of its contributors may be used to
//       endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================

#include <ros/ros.h>

#include <hector_hazmat_detection/hazmat_detection.h>
#include <hector_worldmodel_msgs/ImagePercept.h>
#include <hector_perception_msgs/PerceptionDataArray.h>

#include <cv.h>
#include <cv_bridge/cv_bridge.h>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include <tpofinder/detect.h>
#include <tpofinder/provide.h>
#include <tpofinder/visualize.h>

#include <ros/package.h>

#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include <ros/package.h>

using namespace cv;
using namespace tpofinder;
using namespace std;

namespace hector_hazmat_detection {

  struct RosDebugFilter : public DetectionFilter {

    RosDebugFilter(const cv::Ptr<DetectionFilter> filter)
    /*    */ : filter_(filter){
  }

  virtual bool accept(const Detection & d){
    ROS_INFO("Filter: Inliers %d, Matches %d, Ratio: %f", d.inliers.size(), d.matches.size(), (d.inliers.size() / (double) d.matches.size()));
    return filter_->accept(d);
  }

private:
  cv::Ptr<DetectionFilter> filter_;

};

struct HomographyFilter : public DetectionFilter {

  HomographyFilter(){
  }

  virtual bool accept(const Detection & d){
    double det = d.homography.at<double>(0,0)*d.homography.at<double>(1,1)-d.homography.at<double>(0,1)*d.homography.at<double>(1,0);
    ROS_INFO("DET: %f", det);
    return det > 0.0;
  }

};

struct NumberInliersFilter : public DetectionFilter {

  NumberInliersFilter(int threshold)
  /*    */ : threshold_(threshold){
}

virtual bool accept(const Detection & d){
  return d.inliers.size() > threshold_;
}

private:
  int threshold_;

};

hazmat_detection_impl::hazmat_detection_impl(ros::NodeHandle nh, ros::NodeHandle priv_nh)
: nh_(nh)
, image_transport_(nh_)
, listener_(0)
, debug_provider_(nh_)
{

  detection_output_folder_ = ros::package::getPath("hector_hazmat_detection")+"/detection_debug"; //TODO as parameter
  perceptClassId_ = "hazmat_sign"; //TODO as parameter

  rotation_image_size_ = 2;
  priv_nh.getParam("rotation_enabled", rotation_enabled);
  priv_nh.getParam("rotation_source_frame", rotation_source_frame_id_);
  priv_nh.getParam("rotation_target_frame", rotation_target_frame_id_);
  priv_nh.getParam("rotation_image_size", rotation_image_size_);

  worldmodel_percept_publisher_ = nh_.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 10);
  aggregator_percept_publisher_ = nh_.advertise<hector_perception_msgs::PerceptionDataArray>("perception/image_percept", 10);


  hazmat_image_publisher_ = image_transport_.advertiseCamera("image/hazmat", 10);
  camera_subscriber_ = image_transport_.subscribeCamera("image", 50, &hazmat_detection_impl::imageCallback, this);

  if (!rotation_target_frame_id_.empty()) {
    listener_ = new tf::TransformListener();
    rotated_image_publisher_ = image_transport_.advertiseCamera("image/rotated", 10);
  }

  ROS_INFO("Setting up tpofinder");

  Ptr<FeatureDetector> fd = new OrbFeatureDetector(1000, 1.2, 8);
  Ptr<FeatureDetector> trainFd = new OrbFeatureDetector(250, 1.2, 8);
  Ptr<DescriptorExtractor> de = new OrbDescriptorExtractor(1000, 1.2, 8);

  Ptr<flann::IndexParams> indexParams = new flann::LshIndexParams(15, 12, 2);
  Ptr<DescriptorMatcher> dm = new FlannBasedMatcher(indexParams);

  Feature trainFeature(trainFd, de, dm);

  Modelbase modelbase(trainFeature);

  ROS_INFO("Loading Training Data");

  std::vector<string> models;
  priv_nh.getParam("models", models);
  for(unsigned i=0; i < models.size(); i++) {
    ROS_INFO("Loading model %s", models[i].c_str());
    loadModel(modelbase, models[i]);
  }

  if(models.size() < 1){
    ROS_ERROR("No models specified");

  }

  ROS_INFO("Creating Detector");

  Feature feature(fd, de, dm);

  Ptr<DetectionFilter> filter = new RosDebugFilter(new AndFilter(new NumberInliersFilter(100), new HomographyFilter())
  //     new AndFilter(new MagicHomographyFilter(), new NumberInliersFilter(15))
);
/*
new AndFilter(
Ptr<DetectionFilter> (new EigenvalueFilter(-1, 4.0)),
Ptr<DetectionFilter> (new InliersRatioFilter(0.30))));*/


detector = new Detector (modelbase, feature, filter);

ROS_INFO("Successfully initialized the hazmat detector for image %s", camera_subscriber_.getTopic().c_str());
}

hazmat_detection_impl::~hazmat_detection_impl()
{
  delete listener_;
}

void hazmat_detection_impl::saveDetection(const Detection& detection,const Mat& processingImage,const Mat& detectionImage){
  boost::filesystem::path dir(detection_output_folder_);
  std::string folder_name =boost::posix_time::to_iso_string(boost::posix_time::microsec_clock::local_time());

  dir /= folder_name;

  ROS_WARN("Directory: %s", dir.c_str());

  if(boost::filesystem::create_directories(dir)) {

    imwrite((dir / "processingImage.jpg").c_str(), processingImage);
    imwrite((dir / "detectionImage.jpg").c_str(), detectionImage);

    std::ofstream outfile((dir / "detection.txt").c_str());

    if(!outfile.is_open()) {
      ROS_ERROR("Couldn't open 'detection.txt'");
      return;
    }

    outfile << "Inliers: " << detection.inliers.size() << std::endl;
    outfile << "Matches: " << detection.matches.size() << std::endl;
    outfile.close();
  }else{
    ROS_ERROR("Unable to create output folder");
  }

}

void hazmat_detection_impl::loadModel(Modelbase& modelbase, const string& path) {
  ROS_INFO("Loading object %s", path.c_str());
  modelbase.add(path);
}

void hazmat_detection_impl::rotate_image(cv_bridge::CvImageConstPtr& cv_image, const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info)
{

  cv::Mat rotation_matrix = cv::Mat::eye(2,3,CV_32FC1);
  double rotation_angle = 0.0;

  ROS_DEBUG("Received new image with %u x %u pixels.", image->width, image->height);

  if (!rotation_target_frame_id_.empty() && listener_) {
    tf::StampedTransform transform;
    std::string source_frame_id_ = rotation_source_frame_id_.empty() ? image->header.frame_id : rotation_source_frame_id_;
    try
    {
      listener_->waitForTransform(rotation_target_frame_id_, source_frame_id_, image->header.stamp, ros::Duration(1.0));
      listener_->lookupTransform(rotation_target_frame_id_, source_frame_id_, image->header.stamp, transform);
    } catch (tf::TransformException& e) {
      ROS_ERROR("%s", e.what());
      return;
    }

    // calculate rotation angle
    tfScalar roll, pitch, yaw;
    transform.getBasis().getRPY(roll, pitch, yaw);
    rotation_angle = -roll;

    // Transform the image.
    try
    {
      cv::Mat in_image = cv_image->image;

      // Compute the output image size.
      int max_dim = in_image.cols > in_image.rows ? in_image.cols : in_image.rows;
      int min_dim = in_image.cols < in_image.rows ? in_image.cols : in_image.rows;
      int noblack_dim = min_dim / sqrt(2);
      int diag_dim = sqrt(in_image.cols*in_image.cols + in_image.rows*in_image.rows);
      int out_size;
      int candidates[] = { noblack_dim, min_dim, max_dim, diag_dim, diag_dim }; // diag_dim repeated to simplify limit case.
      int step = rotation_image_size_;
      out_size = candidates[step] + (candidates[step + 1] - candidates[step]) * (rotation_image_size_ - step);
      //ROS_INFO("out_size: %d", out_size);

      // Compute the rotation matrix.
      rotation_matrix = cv::getRotationMatrix2D(cv::Point2f(in_image.cols / 2.0, in_image.rows / 2.0), 180 * rotation_angle / M_PI, 1);
      rotation_matrix.at<double>(0, 2) += (out_size - in_image.cols) / 2.0;
      rotation_matrix.at<double>(1, 2) += (out_size - in_image.rows) / 2.0;

      // Do the rotation
      cv_bridge::CvImage *temp = new cv_bridge::CvImage(*cv_image);
      cv::warpAffine(in_image, temp->image, rotation_matrix, cv::Size(out_size, out_size));
      cv_image.reset(temp);

      debug_provider_.addDebugImage(cv_image->image);

      if (rotated_image_publisher_.getNumSubscribers() > 0) {
        sensor_msgs::Image rotated_image;
        cv_image->toImageMsg(rotated_image);
        rotated_image_publisher_.publish(rotated_image, *camera_info);
      }
    }
    catch (cv::Exception &e)
    {
      ROS_ERROR("Image processing error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
      return;
    }
  }
}

void hazmat_detection_impl::publishDetection(const Detection& detection){

}

void hazmat_detection_impl::imageCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info)
{
  clock_t start = clock();
  cv_bridge::CvImageConstPtr cv_image;
  cv_image = cv_bridge::toCvShare(image, "bgr8");

  debug_provider_.addDebugImage(cv_image->image);

  if(rotation_enabled){
    rotate_image(cv_image, image, camera_info);
  }

  Mat detectionImage = cv_image->image.clone();

  for(int i = 0; i < 4; i++){
    cv::Mat cropped_image = cv_image->image.clone();

    if(i == 0){

      rectangle(cropped_image, Point(0,0), Point(640,190), Scalar( 0, 0, 0 ), -1); //h1
      rectangle(cropped_image, Point(370,0), Point(640,480), Scalar( 0, 0, 0 ), -1); //v1

    }else if( i == 1){
      rectangle(cropped_image, Point(0,290), Point(640,480), Scalar( 0, 0, 0 ), -1); //h2
      rectangle(cropped_image, Point(370,0), Point(640,480), Scalar( 0, 0, 0 ), -1); //v1
    }else if(i == 2){
      rectangle(cropped_image, Point(0,290), Point(640,480), Scalar( 0, 0, 0 ), -1); //h2
      rectangle(cropped_image, Point(0,0), Point(270,480), Scalar( 0, 0, 0 ), -1); //v2
    }else {
      rectangle(cropped_image, Point(0,0), Point(640,190), Scalar( 0, 0, 0 ), -1); //h1
      rectangle(cropped_image, Point(0,0), Point(270,480), Scalar( 0, 0, 0 ), -1); //v2
    }

    debug_provider_.addDebugImage(cropped_image);

    vector<Detection> detections;
    int trys = 3;

    int detectionCount = 0;

    Mat processingImage = cropped_image.clone();


    hector_perception_msgs::PerceptionDataArray perception_array;
    perception_array.header = image->header;
    perception_array.perceptionType = "hazmat";


    do{
      Mat currentDetectionImage = cv_image->image.clone();
      Scene scene = detector->describe(processingImage);

      Mat keypoint_image;
      drawKeypoints(cv_image->image, scene.keypoints, keypoint_image);

    //  debug_provider_.addDebugImage(keypoint_image);

      detections = detector->detect(scene);

      Mat detectedObjects = Mat::zeros(cv_image->image.rows, cv_image->image.cols, CV_8U);

      ROS_INFO("Found %d objects in image", detections.size());
      int o_i = 1;

      BOOST_FOREACH(Detection d, detections) {
        ROS_INFO("Object %d", o_i);
        ROS_INFO("Name: %s", d.model.name.c_str());
        ROS_DEBUG("drawing detection into overall image");
        drawDetection(detectionImage, d);
        ROS_DEBUG("drawing detection into current image");
        drawDetection(currentDetectionImage, d);
        ROS_DEBUG("saving detection for debug");
        saveDetection(d, processingImage, currentDetectionImage);
        o_i++;

        ROS_DEBUG("Warping model for ROI");
        Mat tRoi;
        // TODO: Is this the correct size? Need to test this with a model image
        // smaller or larger than the scene image
        warpPerspective(d.model.views[0].roi, tRoi, d.homography,
          d.model.views[0].image.size());

          SimpleBlobDetector::Params params;
          params.blobColor = 255;

          SimpleBlobDetector detector(params);

          // Detect blobs.
          std::vector<KeyPoint> keypoints;
          ROS_INFO("Detecting blob");
          Mat blob_image = tRoi.clone();
          detector.detect( blob_image , keypoints);

          KeyPoint center;

          if(keypoints.size()>1){
            ROS_WARN("Number of keypoints too large for finding detection. Should be 1");
            center = keypoints[0];
          }else if (keypoints.size()<1){
            ROS_WARN("No detection found in ROI");
            //TODO select one point
          }else{
            center = keypoints[0];
          }

          ROS_DEBUG("publishing image percept");
          hector_worldmodel_msgs::ImagePercept ip;
          ip.header= image->header;
          ip.info.class_id = perceptClassId_;
          ip.info.class_support = 1;
          ip.camera_info = *camera_info;

          ip.x = center.pt.x;
          ip.y = center.pt.y;

          ROS_INFO("Image percept: %d %d", ip.x, ip.y);

          ip.info.class_id = d.model.name;

          worldmodel_percept_publisher_.publish(ip);

          hector_perception_msgs::PerceptionData perception_data;
          perception_data.percept_name = perceptClassId_;
          geometry_msgs::Polygon polygon;
          geometry_msgs::Point32 p0,p1,p2,p3;
          //todo add correct size
          p0.x = center.pt.x+25;
          p0.y = center.pt.y+25;
          p1.x = center.pt.x+25;
          p1.y = center.pt.y-25;
          p2.x = center.pt.x-25;
          p2.y = center.pt.y-25;
          p3.x = center.pt.x-25;
          p3.y = center.pt.y+25;
          polygon.points.push_back(p0);
          polygon.points.push_back(p1);
          polygon.points.push_back(p2);
          polygon.points.push_back(p3);
          perception_data.polygon = polygon;
          perception_array.perceptionList.push_back(perception_data);

          ROS_DEBUG("Updating detected objects in current image");
          if(detectedObjects.rows == tRoi.rows && detectedObjects.cols == tRoi.cols && detectedObjects.type() == tRoi.type()){
            detectedObjects += tRoi;
          }else{
            ROS_WARN("Size missmatch for ROI. Will not add detected object");
            ROS_WARN("objects %d %d %d", detectedObjects.rows, detectedObjects.cols, detectedObjects.type());
            ROS_WARN("tRoi %d %d %d", tRoi.rows, tRoi.cols, tRoi.type());
          }

        }
        detectionCount += detections.size();

        cvtColor(detectedObjects,detectedObjects,CV_GRAY2RGB);

        //debug_provider_.addDebugImage(detectedObjects);
        ROS_DEBUG("Removing detected objects from current image for rerun");
        processingImage = processingImage - detectedObjects;
        //debug_provider_.addDebugImage(processingImage);


        trys--;

      }while(detections.size() > 0 && trys > 0);

      if (aggregator_percept_publisher_.getNumSubscribers() > 0)
      {
        aggregator_percept_publisher_.publish(perception_array);
      }

      if(detectionCount > 0){
        break;
      }

      //TODO if found, skip rest

    }

    debug_provider_.addDebugImage(detectionImage);
    debug_provider_.publishDebugImage();
    clock_t ticks = clock()-start;
    ROS_INFO("Detection time: %f", (double)ticks/CLOCKS_PER_SEC);

  }

} // namespace hector_qrcode_detection
