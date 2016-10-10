/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2013, PAL Robotics, S.L.
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
 * @author Jordi Pages
 */
#ifndef _PAL_TEXTURE_DETECTOR_H_
#define _PAL_TEXTURE_DETECTOR_H_

// OpenCV headers
#include <opencv2/core/core.hpp>

// Boost headers
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

// Std C++ headers
#include <memory>
#include <string>
#include <vector>

namespace cv {

  class Mat;

  template<typename _Tp>
  class Rect_;

  template<typename _Tp>
  class Point_;

  template<typename _Tp>
  class Size_;

}

namespace pal {

  class TextureDetectorImpl;

  /**
   * @brief The TextureDetector
   */
  class TextureDetector: boost::noncopyable {

  public:

    /**
     * @brief TextureDetector create a texture detector with the given object that will be scaled at different levels
     * @param targetObject image of the object to be detected
     * @param scales a vector with the scales at which the target object will be considered. If empty, the target object
     *        won't be scaled
     * @param useRatioTest
     * @param estimateHomography
     * @param homographyIterations,
     * @param showDebug
     */
    TextureDetector(const cv::Mat& targetObject,
                    const std::vector<double>& scales,
                    bool useRatioTest = true,
                    bool estimateHomography = true,
                    int homographyIterations = 1,
                    bool showDebug = false);

    virtual ~TextureDetector();

    /**
     * @brief setTargetObject change the image of the object to find
     * @param targetObject
     */
    void setTargetObject(const cv::Mat& targetObject);


    bool detect(const cv::Mat& img,
                std::vector< cv::Point_<float> >& roi);

    /**
     * @brief detect
     * @param img
     * @param objectSize the width and height of the object in meters
     * @param cameraMatrix 3x3 matrix of intrinsic parameters
     * @param distCoeff distortion coefficients. It must be a 1x4, 1x5 or 1x8 vector
     * @param[out] roi
     * @param[out] pose 4x4 matrix expressing the pose of the object frame in the camera frame.
     *             The object frame is placed in the centre of it with X pointing downwards,
     *             Y rightwards and Z backwards the object plane wrt to camera.
     * @return
     */
    bool detect(const cv::Mat& img,
                const cv::Size_<float>& objectSize,
                const cv::Mat& cameraMatrix,
                const cv::Mat& distCoeff,
                std::vector< cv::Point2f >& roi,
                cv::Mat& pose);

    static void drawRoi(const std::vector< cv::Point2f >& roi,
                        cv::Mat& img);

  protected:

    boost::scoped_ptr<TextureDetectorImpl> _impl;

  };

}

#endif //_PAL_TEXTURE_DETECTOR_H_
