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

#include <hector_hazmat_detection/hazmat_detection.h>
#include <hector_worldmodel_msgs/ImagePercept.h>

#include <cv.h>
#include <cv_bridge/cv_bridge.h>

namespace hector_hazmat_detection {

hazmat_detection_impl::hazmat_detection_impl(ros::NodeHandle nh, ros::NodeHandle priv_nh)
  : nh_(nh)
  , image_transport_(nh_)
  , listener_(0)
{

  rotation_image_size_ = 2;
  priv_nh.getParam("rotation_source_frame", rotation_source_frame_id_);
  priv_nh.getParam("rotation_target_frame", rotation_target_frame_id_);
  priv_nh.getParam("rotation_image_size", rotation_image_size_);

  percept_publisher_ = nh_.advertise<hector_worldmodel_msgs::ImagePercept>("image_percept", 10);
  hazmat_image_publisher_ = image_transport_.advertiseCamera("image/hazmat", 10);
  camera_subscriber_ = image_transport_.subscribeCamera("image", 10, &hazmat_detection_impl::imageCallback, this);

  if (!rotation_target_frame_id_.empty()) {
    listener_ = new tf::TransformListener();
    rotated_image_publisher_ = image_transport_.advertiseCamera("image/rotated", 10);
  }

  ROS_INFO("Successfully initialized the hazmat detector for image %s", camera_subscriber_.getTopic().c_str());
}

hazmat_detection_impl::~hazmat_detection_impl()
{
  delete listener_;
}

void hazmat_detection_impl::imageCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& camera_info)
{

}

} // namespace hector_qrcode_detection
