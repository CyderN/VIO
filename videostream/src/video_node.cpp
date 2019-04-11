#include <queue>
#include <sensor_msgs/Imu.h>
#include <ros/forwards.h>
#include <sensor_msgs/Image.h>

#include <tf/transform_broadcaster.h>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <mutex>
#include <condition_variable>
#include <thread>
using namespace cv;
using namespace std;

typedef std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::ImageConstPtr>> mymeasurements;

std::mutex measurements_mutex;
std::condition_variable cond;

std::queue<sensor_msgs::ImuConstPtr> imu_queue;
std::queue<sensor_msgs::ImageConstPtr> image_queue;


class featurept
{
public:
    cv::KeyPoint keypoint;
    std::vector<cv::KeyPoint> dangerpoints;
    bool indanger = 0;
    int dangernum;
};

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    measurements_mutex.lock();
    imu_queue.push(imu_msg);
    measurements_mutex.unlock();
    //cond.notify_one();
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    measurements_mutex.lock();
    image_queue.push(image_msg);
    measurements_mutex.unlock();
    if(image_queue.size()>=2)
    {
        ROS_INFO("successfully push image");
        cond.notify_one();
    }



}

mymeasurements getmeasurements()
{
    mymeasurements measurements;
    while (true)
    {
        if (imu_queue.empty() || image_queue.empty())
        {
            ROS_WARN("opps: IMU or Image queue empty!");
            return measurements;
        }
        if (!(imu_queue.back()->header.stamp.toSec() > image_queue.front()->header.stamp.toSec() + 0.0))
        {
            ROS_WARN("wait for imu, only should happen at the beginning");
            return measurements;
        }

        if (!(imu_queue.front()->header.stamp.toSec() < image_queue.front()->header.stamp.toSec() + 0.0))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            image_queue.pop();
            continue;
        }
        sensor_msgs::ImageConstPtr img_msg = image_queue.front();
        image_queue.pop();
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_queue.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + 0.0)
        {
            IMUs.emplace_back(imu_queue.front());
            imu_queue.pop();
        }
        IMUs.emplace_back(imu_queue.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
        ROS_INFO("haha: pop measurements successfully~");
    }
    return measurements;
}


int computeHammingDistance(int i, int j, Mat descriptors_1)
{
    int result = 0;
    HammingLUT lut;
    result += lut(&descriptors_1.at<uchar>(i,0), &descriptors_1.at<uchar>(j,0),32);
    return result;
}

void find_danger_points(const Mat& img_1, std::vector<featurept>& featurepts){
    ROS_DEBUG("------------begin find danger points----------");
    featurepts.clear();
    ROS_DEBUG("------------clear last danger-point vector----------");
    Mat descriptors_1;
    std::vector<KeyPoint> key_points_1;
    featurept temp_featurept;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    ROS_DEBUG("------------creat ORB front-end finished----------");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, key_points_1);
    ROS_DEBUG("------------detect corner finished----------");
    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, key_points_1, descriptors_1 );
    ROS_DEBUG("------------compute BRIEF finished----------");
    if(key_points_1.size()<3)return;
    //-- 第三步:将本帧的所有特征点描述子相互异或和，将异或和低于30的作为危险匹配，（描述子256位）
    int counter_j = 0;
    for(auto kpt : key_points_1)
    {
        counter_j++;
        //initial the temp_featurept
        temp_featurept.keypoint = kpt;
        temp_featurept.dangerpoints.clear();
        temp_featurept.indanger = false;
        temp_featurept.dangernum = 0;
        //find danger feature
        int counter_i = 0;
        for(auto kptarget : key_points_1)
        {
            counter_i++;
            if ( (kpt.pt.x==kptarget.pt.x)&&(kpt.pt.y==kptarget.pt.y) )// see if it is the same k-point.
                continue;
            if ( computeHammingDistance(counter_i,counter_j,descriptors_1)<= 60 )
            {
                temp_featurept.dangerpoints.push_back(kptarget);
                temp_featurept.indanger = true;
                temp_featurept.dangernum++;
            }
        }
        //cout<<"dangernum="<<temp_featurept.dangernum<<endl;
        featurepts.push_back(temp_featurept);
    }
    /*
    Mat out_image;
    drawKeypoints(img_1,featurepts.at(1).dangerpoints,out_image, Scalar(0,255,0),2);
    circle(out_image, featurepts.at(1).keypoint.pt, 10, Scalar(255, 0, 0),2);
    for(auto ppts : featurepts.at(1).dangerpoints)
        line(out_image, featurepts.at(1).keypoint.pt, ppts.pt, Scalar(0, 0, 255), 2);
    cv::imshow("opps", out_image);
    cv::waitKey(3);
    */
}

Mat rosImageToCvMat(sensor_msgs::ImageConstPtr image){
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    Mat img_1;
    static float K[3][3] = {354.83758544921875000,0,   328.50021362304687500,
                            0, 354.83068847656250000,  240.57057189941406250,
                            0,                     0,        1};
    static float KK[4] = {-0.29249572753906250, 0.07487106323242188, -0.00019836425781250, -0.00031661987304688};
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_32FC1,K);
    cv::Mat distortion_coefficients = cv::Mat(4,1,CV_32FC1,KK);
    undistort(cv_ptr->image, img_1, camera_matrix, distortion_coefficients);
    return img_1;
}

void track_danger_points(const Mat& temp_mat, std::vector<featurept>& featurepts, const Mat& next_mat)
{
    //for every key-points in the frame.
    //ROS_INFO("-------for every key-points----------");
    for(auto fpts = featurepts.begin(); fpts < featurepts.begin()+1; ++fpts )
    {
        vector<cv::Point2f> temp_pts;
        vector<cv::Point2f> next_pts;
        vector<uchar> status;
        vector<float> err;
        //ROS_INFO("-------for every danger-points-----");
        for(auto dangerpts : fpts->dangerpoints)
        {
            temp_pts.push_back(dangerpts.pt);
        }
        //ROS_INFO("Calculates an optical flow for danger points");
        if (temp_pts.size() > 0){
            cv::calcOpticalFlowPyrLK(temp_mat, next_mat, temp_pts, next_pts, status, err, cv::Size(21, 21), 0);

            Mat out_image;
            drawKeypoints(temp_mat, fpts->dangerpoints, out_image, Scalar(0,255,0),2);
            circle(out_image, fpts->keypoint.pt, 10, Scalar(255, 0, 0),2);
            for(auto ppts : fpts->dangerpoints)
                line(out_image, fpts->keypoint.pt, ppts.pt, Scalar(0, 0, 255), 2);
            int i = 0;
            for(auto tpt : temp_pts)
            {
                if(status[i]=1)line(out_image, tpt, next_pts[i], Scalar(0, 255, 255), 4);
                i++;
            }
            cv::imshow("track_optical", out_image);
            cv::waitKey(3);

        }

        //ROS_INFO("Finish calculating an optical flow for danger points");
        //Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
    }

}

void processmeasurements(mymeasurements &measurements)
{
    std::vector<featurept> featurepts;
    if(measurements.size()<2)
    {
        ROS_ERROR("measurement is in small size!");
        ROS_ERROR("%d",int(measurements.size()));
        return;
    }
    for(mymeasurements::iterator measurement = measurements.begin(); measurement<measurements.end()-1; ++measurement )
    {
        //ROS_ERROR("step3");
        Mat temp_mat = rosImageToCvMat(measurement->second);
        //ROS_ERROR("step4");
        find_danger_points(temp_mat, featurepts);
        //ROS_ERROR("step0");
        Mat next_mat = rosImageToCvMat( (measurement+1)->second );
        //ROS_ERROR("step2");
        //to track all danger-points between two frames.
        track_danger_points(temp_mat, featurepts, next_mat);
        //ROS_ERROR("step1");
    }
    //ROS_INFO("finish track danger points");
}



void process(){
    while(true)
    {
        std::unique_lock<std::mutex> locker(measurements_mutex);
        cond.wait(locker);
        mymeasurements measurements = getmeasurements();
        processmeasurements(measurements);
        locker.unlock();
    }

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_converter");
    ros::NodeHandle nh_;
    ros::Subscriber imu_sub = nh_.subscribe("/mynteye/imu/data_raw", 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber image_sub = nh_.subscribe("/mynteye/left/image_color", 2000, image_callback);
    //std::thread measurement_process{process};
    ros::spin();
}