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

#ifndef HECTOR_HAZMAT_DETECTION_H
#define HECTOR_HAZMAT_DETECTION_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>

//TODO sort out stuff
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

//#include <hector_tpofinder/configure.h>
#include <tpofinder/detect.h>
#include <tpofinder/provide.h>
#include <tpofinder/visualize.h>
#include <cv_debug_provider/cv_debug_provider.h>
#include <vector>

using namespace cv;
using namespace tpofinder;
using namespace std;

namespace hector_hazmat_detection {

class hazmat_detection_impl {
public:
  hazmat_detection_impl(ros::NodeHandle nh, ros::NodeHandle priv_nh);
  ~hazmat_detection_impl();

protected:
  void rotate_image(cv_bridge::CvImageConstPtr& cv_image, const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info);
  void imageCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info);
  void loadModel(Modelbase& modelbase, const string& path);
  void saveDetection(const Detection& detection, const Mat &processingImage, const Mat &detectionImage);
  void publishDetection(const Detection& detection);
private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport image_transport_;
  image_transport::CameraSubscriber camera_subscriber_;
  image_transport::CameraPublisher rotated_image_publisher_;
  image_transport::CameraPublisher hazmat_image_publisher_;

  ros::Publisher worldmodel_percept_publisher_;
  ros::Publisher aggregator_percept_publisher_;
  std::string perceptClassId_;

  bool rotation_enabled = false;
  tf::TransformListener *listener_;
  std::string rotation_source_frame_id_;
  std::string rotation_target_frame_id_;
  int rotation_image_size_;

  Ptr<Detector> detector;
  CvDebugProvider debug_provider_;

  std::string detection_output_folder_;

};

} // namespace hector_hazmat_detection

#endif // HECTOR_HAZMAT_DETECTION_H
