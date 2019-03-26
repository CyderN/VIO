#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <opencv2/opencv.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <mutex>

std::mutex mtx;

static const std::string OPENCV_WINDOW = "Image window";

using namespace std;
using namespace cv;
int num_img = 0;

float K[3][3] = {712.80450439453125000,0,   656.15618896484375000,
                 0, 712.78521728515625000,  359.16461181640625000,
                 0,                     0,        1};
Mat camera_matrix = Mat(3, 3, CV_32FC1,K);


float KK[4] = {-0.30120468139648438, 0.08044815063476562, -0.00009918212890625, 0.00059127807617188};
Mat distortion_coefficients = Mat(4,1,CV_32FC1,KK);


void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    if(keypoints_1.size()<3||keypoints_2.size()<3)return;

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ))
        {
            matches.push_back ( match[i] );
        }
    }

    Mat img_match;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, img_match);
    //imwrite( "left"+to_string(num_img)+".png", img_1);
    //imwrite( "right"+to_string(num_img)+".png", img_2);
    num_img++;
    cv::waitKey(3);
}




void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1,
                            std::vector<KeyPoint> keypoints_2,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t )
{
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算本质矩阵
    Point2d principal_point ( 656.15618896484375000, 359.16461181640625000 );	//相机光心, TUM dataset标定值
    double focal_length = 712.8;			//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
    static tf::TransformBroadcaster br;
    tf::Transform transform(
            tf::Matrix3x3(R.at<double>(0),R.at<double>(1),R.at<double>(2),
                          R.at<double>(3),R.at<double>(4),R.at<double>(5),
                          R.at<double>(6),R.at<double>(7),R.at<double>(8)),
            tf::Vector3(t.at<double>(0),t.at<double>(1),t.at<double>(2))
            );
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "right", "left"));
}






class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_left;
  image_transport::Subscriber image_sub_right;
  image_transport::Publisher image_pub_;

public:
    cv::Mat img_1;
    cv::Mat img_2;
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_left = it_.subscribe("/mynteye/left/image_color", 1,
      &ImageConverter::imageCb1, this);
    image_sub_right = it_.subscribe("/mynteye/right/image_color", 1,
      &ImageConverter::imageCb2, this);

    image_pub_ = it_.advertise("/image_converter/output_video", 1);



    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb1(const sensor_msgs::ImageConstPtr& msg)////////////////一号存图全局 二号匹配一号
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    undistort(cv_ptr->image, img_1, camera_matrix, distortion_coefficients);

  }
    void imageCb2(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        undistort(cv_ptr->image, img_2, camera_matrix, distortion_coefficients);

    }
};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");







  ImageConverter ic;
  ros::Rate loop_rate(1);
  while(ros::ok())
  {
      mtx.lock();
      vector<KeyPoint> keypoints_1, keypoints_2;
      vector<DMatch> matches;
      cout<<"begin match"<<endl;
      find_feature_matches ( ic.img_1, ic.img_2, keypoints_1, keypoints_2, matches );
      mtx.unlock();

      if(keypoints_1.size()>3&&keypoints_2.size()>3)
      {cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
      cout<<"end match"<<endl;
      //-- 估计两张图像间运动
      cv::Mat R,t;
      if(matches.size() > 10)
          pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );}
      ros::spinOnce();
      loop_rate.sleep();
  }

}
