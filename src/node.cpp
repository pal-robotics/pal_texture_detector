/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2016, PAL Robotics, S.L.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  @author Jordi Pages
 */

// PAL headers
#include <pal_texture_detector/texture_detector.h>
#include <pal_detection_msgs/TexturedObjectDetection.h>

// ROS headers
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <bullet/LinearMath/btQuaternion.h>
#include <bullet/LinearMath/btMatrix3x3.h>
#pragma GCC diagnostic warning "-Wunused-parameter"

// OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost headers
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/filesystem.hpp>

// Std C++ headers
#include <string>
#include <memory>
#include <stdexcept>

class TextureDetectorNode {

public:

  TextureDetectorNode(ros::NodeHandle& nh, ros::NodeHandle& pnh,
                      const std::string& targetImagePath);

  virtual ~TextureDetectorNode();

protected:

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);

  void publish(cv::Mat& img,
               const std::string& image_encoding,
               const std::vector<cv::Point2f>& roi);

  void publishDetection(cv::Mat& img,
                        const std::vector<cv::Point2f>& roi,
                        const ros::Time& timeStamp);

  void publishPose(const ros::Time& timeStamp);

  void publishDebug(cv::Mat& img,
                    const std::string& image_encoding,
                    const std::vector<cv::Point2f>& roi,
                    const ros::Time& timeStamp);

  void convert(cv::Mat& img, sensor_msgs::CompressedImage& compressedImg);

  void getCameraIntrinsics();

  ros::NodeHandle _nh, _pnh;

  cv_bridge::CvImage _cvImg;

  cv::Mat _targetImage;
  bool _estimatePose;
  cv::Size2f _objectSize;
  cv::Mat _cameraMatrix, _distCoeff;
  cv::Mat _pose;
  std::string _cameraFrame;

  image_transport::ImageTransport _imageTransport, _privateImageTransport;
  image_transport::Subscriber _imageSub;

  boost::scoped_ptr<pal::TextureDetector> _detector;

  image_transport::Publisher _imDebugPub;

  ros::Publisher _pub;
  ros::Publisher _posePub;
};

TextureDetectorNode::TextureDetectorNode(ros::NodeHandle& nh,
                                         ros::NodeHandle& pnh,
                                         const std::string& targetImagePath):
  _nh(nh),
  _pnh(pnh),
  _targetImage( cv::imread(targetImagePath) ),
  _estimatePose(true),
  _objectSize(0.1445,0.184),
  _imageTransport(nh),
  _privateImageTransport(pnh)
{
  bool enableRatioTest      = true;
  bool enableHomography     = true;
  int  homographyIterations = 2;
  double objectWidth(0), objectHeight(0);
  bool showDebugImages      = false;
  _pnh.getParam("enable_ratio_test", enableRatioTest);
  _pnh.getParam("enable_homography", enableHomography);
  _pnh.getParam("homography_iterations", homographyIterations);
  _pnh.getParam("estimate_pose", _estimatePose);
  _pnh.getParam("object_width", objectWidth);
  _pnh.getParam("object_height", objectHeight);
  _objectSize.width  = static_cast<float>(objectWidth);
  _objectSize.height = static_cast<float>(objectHeight);

  ROS_INFO_STREAM("Ratio test enabled:        " << enableRatioTest);
  ROS_INFO_STREAM("Homography enabled:        " << enableHomography);
  if ( enableHomography )
    ROS_INFO_STREAM("Homography iterations:     " << homographyIterations);
  ROS_INFO_STREAM("Pose estimation enabled:   " << _estimatePose);
  if ( _estimatePose )
    ROS_INFO_STREAM("Planar object width:       " << objectWidth << " m  height: " << objectHeight);

  _pnh.getParam("enable_visual_debug", showDebugImages);

  std::vector<double> scales;  
  scales.push_back(1.0);

  _detector.reset(new pal::TextureDetector(_targetImage,
                                           scales,
                                           enableRatioTest,
                                           enableHomography,
                                           homographyIterations,
                                           showDebugImages));

  image_transport::TransportHints transportHint("raw");
  _imageSub   = _imageTransport.subscribe("rectified_image", 1, &TextureDetectorNode::imageCallback, this, transportHint);

  if ( _estimatePose )
    getCameraIntrinsics();

  _pub     = _pnh.advertise<pal_detection_msgs::TexturedObjectDetection>("detection", 1);

  if ( _estimatePose )
    _posePub = _pnh.advertise<geometry_msgs::PoseStamped>("pose", 1);

  _imDebugPub = _privateImageTransport.advertise("debug", 1);

}

TextureDetectorNode::~TextureDetectorNode()
{
}

void TextureDetectorNode::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cvImgPtr;
  cvImgPtr = cv_bridge::toCvShare(msg);

  _cameraFrame = msg->header.frame_id;
  cv::Mat img = cvImgPtr->image.clone();
  
  std::vector<cv::Point2f> roi;

  if ( _estimatePose )
    _detector->detect(img, _objectSize, _cameraMatrix, _distCoeff, roi, _pose);
  else
    _detector->detect(img, roi);
      
  publish(img, cvImgPtr->encoding, roi);
}

void TextureDetectorNode::getCameraIntrinsics()
{
  ROS_INFO("Waiting for camera_info topic ...");
  sensor_msgs::CameraInfoConstPtr msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camera_info");
  ROS_INFO("Ok");


  _cameraMatrix = cv::Mat::zeros(3,3, CV_32FC1);

  //get focal lengths and image centre from P matrix
  _cameraMatrix.at<float>(0,0) =    msg->P[0]; _cameraMatrix.at<float>(0,1) =         0; _cameraMatrix.at<float>(0,2) = msg->P[2];
  _cameraMatrix.at<float>(1,0) =            0; _cameraMatrix.at<float>(1,1) = msg->P[5]; _cameraMatrix.at<float>(1,2) = msg->P[6];
  _cameraMatrix.at<float>(2,0) =            0; _cameraMatrix.at<float>(2,1) =         0; _cameraMatrix.at<float>(2,2) =         1;
  //rectified image has no distortion
  _distCoeff = cv::Mat(1, 4, CV_32F, cv::Scalar(0.00001));
}



void TextureDetectorNode::convert(cv::Mat& img, sensor_msgs::CompressedImage& compressedImg)
{
  if ( img.channels() == 3 && img.depth() == CV_8U )
    compressedImg.format = sensor_msgs::image_encodings::BGR8;

  else if ( img.channels() == 1 && img.depth() == CV_8U )
    compressedImg.format = sensor_msgs::image_encodings::MONO8;
  else
    throw std::runtime_error("Error in TextureDetectorNode::convert: only 24-bit BGR or 8-bit MONO images are currently supported");

  compressedImg.format += "; jpeg compressed";

  //compression settings
  std::vector<int> params;
  params.resize(3, 0);
  params[0] = CV_IMWRITE_JPEG_QUALITY;
  params[1] = 80; //jpeg quality
  if ( !cv::imencode(".jpg", img, compressedImg.data, params) )
    ROS_ERROR("Error in TextureDetectorNode::convert: cv::imencode failed");
}

void TextureDetectorNode::publishDebug(cv::Mat& img,
                                       const std::string& image_encoding,
                                       const std::vector<cv::Point2f>& roi,
                                       const ros::Time& timeStamp)
{
  _cvImg.encoding = image_encoding;

  _cvImg.image = img.clone();

  if ( !roi.empty() )
    pal::TextureDetector::drawRoi(roi, _cvImg.image);

  sensor_msgs::Image imgMsg;
  imgMsg.header.stamp = timeStamp;
  _cvImg.toImageMsg(imgMsg); //copy image data to ROS message

  _imDebugPub.publish(imgMsg);
}

void TextureDetectorNode::publishDetection(cv::Mat& img,
                                           const std::vector<cv::Point2f>& roi,
                                           const ros::Time& timeStamp)
{
  pal_detection_msgs::TexturedObjectDetection msg;
  msg.header.stamp     = timeStamp;
  msg.img.header.stamp = timeStamp;
  msg.roi.header.stamp = timeStamp;
  convert(img, msg.img);

  for (unsigned int i = 0; i < roi.size(); ++i)
  {
    msg.roi.x.push_back( roi[i].x );
    msg.roi.y.push_back( roi[i].y );
  }

  _pub.publish(msg);
}

void TextureDetectorNode::publishPose(const ros::Time& timeStamp)
{
  geometry_msgs::PoseStamped poseMsg;

  btMatrix3x3 rot(_pose.at<float>(0,0), _pose.at<float>(0,1), _pose.at<float>(0,2),
                  _pose.at<float>(1,0), _pose.at<float>(1,1), _pose.at<float>(1,2),
                  _pose.at<float>(2,0), _pose.at<float>(2,1), _pose.at<float>(2,2));
  btQuaternion quat;
  rot.getRotation(quat);
  poseMsg.header.stamp = timeStamp;
  poseMsg.header.frame_id = _cameraFrame;
  poseMsg.pose.position.x = _pose.at<float>(0,3);
  poseMsg.pose.position.y = _pose.at<float>(1,3);
  poseMsg.pose.position.z = _pose.at<float>(2,3);
  poseMsg.pose.orientation.x = quat.getX();
  poseMsg.pose.orientation.y = quat.getY();
  poseMsg.pose.orientation.z = quat.getZ();
  poseMsg.pose.orientation.w = quat.getW();
  _posePub.publish(poseMsg);
}

void TextureDetectorNode::publish(cv::Mat& img,
                                  const std::string& image_encoding,
                                  const std::vector<cv::Point2f>& roi)
{   
  ros::Time now = ros::Time::now();

  if ( _pub.getNumSubscribers() > 0 )
    publishDetection(img, roi, now);

  if ( _imDebugPub.getNumSubscribers() > 0 )
    publishDebug(img, image_encoding, roi, now);

  if ( !roi.empty() && _estimatePose && _posePub.getNumSubscribers() > 0 )
    publishPose(now);
}

int main(int argc, char** argv)
{
  // Init the ROS node
  ros::init(argc, argv, "texture_detector");
 
  // Precondition: Valid clock
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  ros::CallbackQueue cbQueue;
  nh.setCallbackQueue(&cbQueue);


  if (!ros::Time::waitForValid(ros::WallDuration(10.0))) // NOTE: Important when using simulated clock
  {
    ROS_FATAL("Timed-out waiting for valid time.");
    return EXIT_FAILURE;
  }

  ROS_INFO("Starting textured object detector node");

  std::string targetImagePath = argv[1];

  if ( targetImagePath.empty() )
    throw std::runtime_error("argv[1] must be the path to the target image");

  TextureDetectorNode node(nh, pnh, targetImagePath);

  ROS_INFO_STREAM("Spinning to serve callbacks ...");

  double maxRate = 3.0;
  ROS_INFO_STREAM("The node will run at maximum " << maxRate << " Hz");

  ros::Rate rate(maxRate);
  while ( ros::ok() )
  {
    cbQueue.callAvailable();
    rate.sleep();
  }

}
