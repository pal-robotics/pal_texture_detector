/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2014, PAL Robotics, S.L.
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

#include <pal_texture_detector/texture_detector.h>

// OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

// Std C++ headers
#include <iostream>
#include <stdexcept>

namespace pal {

  /// @cond DOXYGEN_IGNORE
  class TextureDetectorImpl {

  public:        

    TextureDetectorImpl(const cv::Mat& targetObject,
                        const std::vector<double>& scales,
                        bool useRatioTest,
                        bool estimateHomography,
                        int homographyIterations,
                        bool showDebug);

    virtual ~TextureDetectorImpl();

    void setTargetObject(const cv::Mat& targetObject);

    bool detect(const cv::Mat& img,
                std::vector< cv::Point2f >& roi);

    bool detect(const cv::Mat& img,
                const cv::Size2f& objectSize,
                const cv::Mat& cameraMatrix,
                const cv::Mat& distCoeff,
                std::vector< cv::Point2f >& roi,
                cv::Mat& pose);

    static void drawRoi(const std::vector< cv::Point2f >& roi,
                        cv::Mat& img);


  protected:

    void getGrayScale(const cv::Mat& img,
                      cv::Mat& imgGray);

    void getDescriptors(const cv::Mat& img,
                        double scale,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors,
                        cv::Mat& imgScaled);

    void getDescriptors(const cv::Mat& img,
                        double scale,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors);

    void getMatches(const cv::Mat& descriptors,
                    cv::Ptr<cv::DescriptorMatcher>& matcher,
                    std::vector<cv::DMatch>& matches);

    bool discardOutliersWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,
                                     const std::vector<cv::KeyPoint>& trainKeypoints,
                                     std::vector<cv::DMatch>& matches,
                                     cv::Mat& homography);

    enum useScaleIdx { USE_ALL_SCALES = 0, USE_PROVIDED_SCALE = 1 };

    bool computeRobustMatches(const cv::Mat& img,
                              useScaleIdx scaleMode,
                              std::vector<cv::KeyPoint>& keypoints,
                              int& bestTargetScaleIdx,
                              std::vector<cv::DMatch>& matches,
                              bool estimateHomography,
                              cv::Mat& homography);

    void computeRoiFromMatches(const std::vector<cv::DMatch>& matches,
                               const std::vector<cv::KeyPoint>& keypoints,
                               std::vector< cv::Point2f >& roi);

    bool warpImageAndDetect(const cv::Mat& imgGray,
                            const cv::Mat& homography,
                            int bestTargetScale,
                            std::vector<cv::DMatch>& matches,
                            cv::Mat& newHomography);


    cv::Ptr<cv::Feature2D> _featureDetector;
    cv::Ptr<cv::Feature2D> _descriptorExtractor;
    std::vector< std::vector<cv::KeyPoint> > _targetKeypoints;
    std::vector< cv::Ptr<cv::DescriptorMatcher> > _descriptorMatcher;
    std::vector< cv::Mat > _targetScaled;
    std::vector<double> _scales;
    bool _useRatioTest;
    bool _estimateHomography;
    int _homographyIterations;
    cv::Mat _homography; //3x3 homography transforming pixels from the pattern image to the query image
    bool _lookingMatchesInWrappedImg;
    bool _showDebug;

  };


  TextureDetectorImpl::TextureDetectorImpl(const cv::Mat& targetObject,
                                           const std::vector<double>& scales,
                                           bool useRatioTest,
                                           bool estimateHomography,
                                           int homographyIterations,
                                           bool showDebug):
    _featureDetector( cv::ORB::create() ),
    _descriptorExtractor( cv::ORB::create() ),
    _scales(scales),
    _useRatioTest(useRatioTest),
    _estimateHomography(estimateHomography),
    _homographyIterations(homographyIterations),
    _lookingMatchesInWrappedImg(false),
    _showDebug(showDebug)
  {
    if ( _scales.empty() )
      _scales.push_back(1.0);

    setTargetObject(targetObject);

  }

  TextureDetectorImpl::~TextureDetectorImpl()
  {

  }

  void TextureDetectorImpl::setTargetObject(const cv::Mat& targetObject)
  {
    cv::Mat imgGray;
    getGrayScale(targetObject, imgGray);

    _targetScaled.clear();

    for (unsigned int i = 0; i < _scales.size(); ++i)
    {
      //scale the image with the target object to each one of the specified scales
      _targetKeypoints.push_back( std::vector<cv::KeyPoint>() );
      cv::Mat imgScaled, descriptors;
      std::vector<cv::Mat> descriptorsCollection;

      getDescriptors(imgGray, _scales[i], _targetKeypoints[i], descriptors, imgScaled);

      _targetScaled.push_back( imgScaled );

      descriptorsCollection.push_back(descriptors);

      //create a descriptor matcher for ech target scale
      //enable cross-match filter only if ratio-test filter is disabled
      _descriptorMatcher.push_back( new cv::BFMatcher(cv::NORM_HAMMING, _useRatioTest ? false : true) );
      _descriptorMatcher[i]->clear();
      _descriptorMatcher[i]->add(descriptorsCollection);
      _descriptorMatcher[i]->train();

      if ( _showDebug )
        std::cout << "Scale: " << _scales[i] << " => " << descriptors.size() << " descriptors added" << std::endl;
    }
  }

  void TextureDetectorImpl::getDescriptors(const cv::Mat& img,
                                           double scale,
                                           std::vector<cv::KeyPoint>& keypoints,
                                           cv::Mat& descriptors,
                                           cv::Mat& imgScaled)
  {
    if ( scale != 1.0 )
      cv::resize(img, imgScaled, cv::Size(0,0), scale, scale);
    else
      imgScaled = img;

    _featureDetector->detect(imgScaled, keypoints);
    _descriptorExtractor->compute(imgScaled, keypoints, descriptors);
  }

  void TextureDetectorImpl::getDescriptors(const cv::Mat& img,
                                           double scale,
                                           std::vector<cv::KeyPoint>& keypoints,
                                           cv::Mat& descriptors)
  {
     cv::Mat imgScaled;

     getDescriptors(img, scale, keypoints, descriptors, imgScaled);
  }

  void TextureDetectorImpl::getMatches(const cv::Mat& descriptors,
                                       cv::Ptr<cv::DescriptorMatcher>& matcher,
                                       std::vector<cv::DMatch>& matches)
  {
    matches.clear();

    if ( _useRatioTest )
    {
      float minRatio = 1.0/1.5;
      std::vector< std::vector<cv::DMatch> > knnMatches;

      matcher->knnMatch(descriptors, knnMatches, 2);

      for (unsigned int i = 0; i < knnMatches.size(); ++i)
      {
        if ( knnMatches[i].size() == 2 )
        {
          if ( fabs(knnMatches[i][1].distance) > 1e-06 )
          {
            float ratio = knnMatches[i][0].distance / knnMatches[i][1].distance;
            if ( ratio < minRatio )
              matches.push_back(knnMatches[i][0]);
          }
        }
      }
    }
    else
    {
      matcher->match(descriptors, matches);
    }
  }

  bool TextureDetectorImpl::discardOutliersWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,
                                                        const std::vector<cv::KeyPoint>& trainKeypoints,
                                                        std::vector<cv::DMatch>& matches,
                                                        cv::Mat& homography)
  {
    const unsigned int MIN_MATCHES = 10;
    const float MAX_REPROJECTION_ERROR = 3;

    if ( matches.size() < MIN_MATCHES )
    {
      if ( _showDebug )
        std::cout << "Not possible to estimate homography as too few points are provided (" <<
                     matches.size() << ")" << std::endl;
      return false;
    }

    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());

    for ( unsigned int i = 0; i < matches.size(); ++i)
    {
      srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
      dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }

    std::vector<unsigned char> inliersMask(srcPoints.size());

    homography = cv::findHomography(srcPoints,
                                    dstPoints,
                                    CV_FM_RANSAC,
                                    MAX_REPROJECTION_ERROR,
                                    inliersMask);

    std::vector<cv::DMatch> inliers;
    for (unsigned int i = 0; i < inliersMask.size(); ++i)
    {
      if ( inliersMask[i] )
        inliers.push_back(matches[i]);
    }

    matches.swap(inliers);
    return matches.size() > MIN_MATCHES;
  }

  bool TextureDetectorImpl::computeRobustMatches(const cv::Mat& img,
                                                 useScaleIdx scaleMode,
                                                 std::vector<cv::KeyPoint>& keypoints,
                                                 int& bestTargetScaleIdx,
                                                 std::vector<cv::DMatch>& matches,
                                                 bool estimateHomography,
                                                 cv::Mat& homography)
  {
    cv::Mat descriptors;
    matches.clear();

    getDescriptors(img, 1.0, keypoints, descriptors);

    unsigned int scaleIdxStart = 0, scaleIdxEnd = _descriptorMatcher.size();

    if ( scaleMode == USE_PROVIDED_SCALE )
    {
      scaleIdxStart = bestTargetScaleIdx;
      scaleIdxEnd   = bestTargetScaleIdx + 1;
    }

    for (unsigned int i = scaleIdxStart; i < scaleIdxEnd; ++i)
    {
      std::vector<cv::DMatch> currentMatches;

      getMatches(descriptors, _descriptorMatcher[i], currentMatches);

      if ( _showDebug )
        std::cout << "Number of matches found at scale " << _scales[i] << ": " << currentMatches.size() << std::endl;

      cv::Mat currentHomography;

      if ( estimateHomography )
      {        
        if ( !discardOutliersWithHomography(keypoints, _targetKeypoints[i], currentMatches, currentHomography) )
        {
          currentMatches.clear();
          if ( _showDebug )
            std::cout << "Homography did not find enough inliers" << std::endl;
        }
        else
        {
          if ( _showDebug )
            std::cout << "Number of matches after homography estimation: " << currentMatches.size() << std::endl;          
        }
      }

      if ( currentMatches.size() > matches.size() )
      {
        matches = currentMatches;
        bestTargetScaleIdx = static_cast<int>(i);
        if ( estimateHomography )
          homography = currentHomography;
      }
    }

    if ( _showDebug && !matches.empty() )
    {
      std::cout << "Best target scale idx: " << bestTargetScaleIdx << " => scale: " << _scales[bestTargetScaleIdx] << std::endl;

      cv::Mat debugImg;
      cv::drawMatches(img, keypoints, _targetScaled[bestTargetScaleIdx], _targetKeypoints[bestTargetScaleIdx], matches, debugImg);

      std::string sufix = " in original image";
      if ( _lookingMatchesInWrappedImg )
        sufix = " in wrapped image";
      cv::imshow("robust matches " + sufix, debugImg);
    }

    return !matches.empty();
  }

  void TextureDetectorImpl::getGrayScale(const cv::Mat& img,
                                         cv::Mat& imgGray)
  {
    if ( img.channels() == 3 )
      cv::cvtColor(img, imgGray, CV_BGR2GRAY);
    else
      imgGray = img.clone();
  }

  bool TextureDetectorImpl::warpImageAndDetect(const cv::Mat& imgGray,
                                               const cv::Mat& homography,
                                               int bestTargetScale,
                                               std::vector<cv::DMatch>& matches,
                                               cv::Mat& newHomography)
  {
    cv::Mat imgGrayWarped;

    cv::warpPerspective(imgGray, imgGrayWarped,
                        homography, _targetScaled[bestTargetScale].size(),
                        cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);

    std::vector<cv::KeyPoint> keypoints;

    return computeRobustMatches(imgGrayWarped,
                                 USE_PROVIDED_SCALE,
                                 keypoints,
                                 bestTargetScale,
                                 matches,
                                 true,
                                 newHomography);
  }

  void TextureDetectorImpl::computeRoiFromMatches(const std::vector<cv::DMatch>& matches,
                                                  const std::vector<cv::KeyPoint>& keypoints,
                                                  std::vector< cv::Point2f >& roi)
  {
    //compute roi as the min area rectangle enclosing all the matches in the query image
    std::vector<cv::Point2f> points(matches.size());

    for ( unsigned int i = 0; i < matches.size(); ++i)
      points[i] = keypoints[matches[i].queryIdx].pt;

    cv::RotatedRect rect = cv::minAreaRect(points);
    cv::Point2f vertices[4];
    rect.points(vertices);
    std::copy(&vertices[0], &vertices[4], std::back_inserter(roi));
  }

  bool TextureDetectorImpl::detect(const cv::Mat& img,
                                   std::vector< cv::Point2f >& roi)
  {
    roi.clear();

    std::vector<cv::DMatch> matches;
    int bestTargetScale;

    cv::Mat imgGray;
    getGrayScale(img, imgGray);

    std::vector<cv::KeyPoint> keypoints;

    if ( _showDebug )
      std::cout << std::endl << "=================" << std::endl;

    //first find matches between the query image and the target image at different scales,
    //using cross-filter or ratio-filter and then homography filtering if enabled
    _lookingMatchesInWrappedImg = false;
    bool found = computeRobustMatches(imgGray, USE_ALL_SCALES,
                                      keypoints, bestTargetScale,
                                      matches, _estimateHomography, _homography);

    if ( !found )
      return false;

    if ( !_estimateHomography )
    {
      computeRoiFromMatches(matches, keypoints, roi);
    }
    else //compute homography
    {
      for (int i = 0; i < _homographyIterations; ++i)
      {
        if ( _showDebug )
          std::cout << "=> Refining homography with warped image" << std::endl;

        cv::Mat refinedHomography;

        _lookingMatchesInWrappedImg = true;
        found = warpImageAndDetect(imgGray, _homography,
                                   bestTargetScale, matches, refinedHomography);

        unsigned int minNumberOfMatches = std::min<unsigned int>(15,
                                                                 _targetKeypoints[bestTargetScale].size()/4);

        if ( !found )
          return false;

        if ( matches.size() < minNumberOfMatches )
        {
          if ( _showDebug )
          {
            std::cout << "Target not detected because a minimum of " <<
                         minNumberOfMatches <<
                         " (should have been found" << std::endl;
          }
          return false;
        }

        _homography = _homography * refinedHomography;
      }

      std::vector<cv::Point2f> patternRoi;

      //compute the target roi in the query image using the homography
      patternRoi.push_back( cv::Point2f(0,0) );
      patternRoi.push_back( cv::Point2f(_targetScaled[bestTargetScale].size().width-1, 0) );
      patternRoi.push_back( cv::Point2f(_targetScaled[bestTargetScale].size().width-1, _targetScaled[bestTargetScale].size().height-1) );
      patternRoi.push_back( cv::Point2f(0, _targetScaled[bestTargetScale].size().height-1) );


      cv::perspectiveTransform(patternRoi, roi, _homography);

      if ( _showDebug )
      {
        cv::Mat imgGrayWarped;
        cv::warpPerspective(imgGray, imgGrayWarped,
                            _homography, _targetScaled[bestTargetScale].size(),
                            cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);
        cv::imshow("final refined warped image", imgGrayWarped);
        cv::waitKey(250);
      }
    }

    return true;
  }

  bool TextureDetectorImpl::detect(const cv::Mat& img,
                                   const cv::Size2f& objectSize,
                                   const cv::Mat& cameraMatrix,
                                   const cv::Mat& distCoeff,
                                   std::vector< cv::Point2f >& roi,
                                   cv::Mat& pose)
  {
    if ( !_estimateHomography )
      throw std::runtime_error("Error in TextureDetectorImpl::detect: unable to estimate the pose if homography estimation is disabled");

    //get the roi of the object in the image and _homography
    bool found = detect(img, roi);

    if ( found )
    {
      std::vector<cv::Point3f> objectPoints;
      objectPoints.push_back( cv::Point3f(-objectSize.width/2, -objectSize.height/2, 0) );
      objectPoints.push_back( cv::Point3f( objectSize.width/2, -objectSize.height/2, 0) );
      objectPoints.push_back( cv::Point3f( objectSize.width/2,  objectSize.height/2, 0) );
      objectPoints.push_back( cv::Point3f(-objectSize.width/2,  objectSize.height/2, 0) );

      cv::Mat rvec, tvec;
      cv::solvePnPRansac(objectPoints, roi,
                   cameraMatrix, distCoeff,
                   rvec, tvec,      //These matrices are returned as CV_64F
                   false, 100, 8.0, 0.9, cv::noArray(), cv::SOLVEPNP_P3P);

      cv::Mat rotation;
      cv::Rodrigues(rvec, rotation);

      pose = cv::Mat::zeros(4,4,CV_32F);
      pose.at<float>(3,3) = 1;
      cv::Mat rotationInPose(pose.colRange(0,3).rowRange(0,3)); //pointer to the top-left most 3x3 part of pose
      rotation.convertTo(rotationInPose, CV_32F);
      cv::Mat translationInPose(pose.colRange(3,4).rowRange(0,3)); //pointer to the top-right most 3x1 part of pose
      tvec.convertTo(translationInPose, CV_32F);
    }

    return found;
  }

  void TextureDetectorImpl::drawRoi(const std::vector< cv::Point2f >& roi,
                                    cv::Mat& img)
  {
    if ( roi.size() == 4 )
    {
      cv::line(img, roi[0], roi[1], CV_RGB(0,255,0), 2);
      cv::line(img, roi[1], roi[2], CV_RGB(0,255,0), 2);
      cv::line(img, roi[2], roi[3], CV_RGB(0,255,0), 2);
      cv::line(img, roi[0], roi[3], CV_RGB(0,255,0), 2);
    }
  }
  /// @endcond

  ///////////////////////////////////////////////////////////////////////////7

  TextureDetector::TextureDetector(const cv::Mat& targetObject,
                                   const std::vector<double>& scales,
                                   bool useRatioTest,
                                   bool estimateHomography,
                                   int homographyIterations,
                                   bool showDebug)
  {
    _impl.reset( new TextureDetectorImpl(targetObject,
                                         scales,
                                         useRatioTest,
                                         estimateHomography,
                                         homographyIterations,
                                         showDebug) );
  }

  TextureDetector::~TextureDetector()
  {
    _impl.reset();
  }

  bool TextureDetector::detect(const cv::Mat& img,
                               std::vector< cv::Point2f >& roi)
  {
    return _impl->detect(img, roi);
  }

  void TextureDetector::setTargetObject(const cv::Mat& targetObject)
  {
    _impl->setTargetObject(targetObject);
  }

  bool TextureDetector::detect(const cv::Mat& img,
                               const cv::Size2f& objectSize,
                               const cv::Mat& cameraMatrix,
                               const cv::Mat& distCoeff,
                               std::vector< cv::Point2f >& roi,
                               cv::Mat& pose)
  {
    return _impl->detect(img,
                         objectSize,
                         cameraMatrix,
                         distCoeff,
                         roi,
                         pose);
  }

  void TextureDetector::drawRoi(const std::vector< cv::Point2f >& roi,
                                cv::Mat& img)
  {
    TextureDetectorImpl::drawRoi(roi, img);
  }

} //pal




