#include "video_node.cpp.h"




int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_converter");
    ImageConverter ic;
    ros::Rate loop_rate(30);
    while(ros::ok())
    {
        vector<featurept> featurepts;
        find_danger_points(ic.img_1, featurepts);
        cacualate_pre_integreation();

        //Mat out_image;
        //drawKeypoints(ic.img_1,featurepts.at(1).dangerpoints,out_image);
        //circle(out_image, featurepts.at(1).keypoint.pt, 2, Scalar(255, 0, 0));
        //cv::imshow("opps", ic.img_1);
        //cv::waitKey();
        cout<<"I am done"<<endl;

        /*
        mtx.lock();
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches;
        cout<<"begin match"<<endl;
        find_feature_matches ( ic.img_1, ic.img_1, keypoints_1, keypoints_2, matches );

        mtx.unlock();

        if(keypoints_1.size()>3&&keypoints_2.size()>3)
        {cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
        cout<<"end match"<<endl;
        //-- 估计两张图像间运动
        cv::Mat R,t;
        if(matches.size() > 10)
          pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );}
        */


        ros::spinOnce();
        loop_rate.sleep();
    }

}
