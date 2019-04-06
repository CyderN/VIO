#include "ImageConverter.h"
#include "IMUreceiver.h"



int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_converter");
    ros::NodeHandle nh_;
    ImageConverter ic(nh_);
    IMUreceiver imurec(nh_);
    ros::Rate loop_rate(30);
    while(ros::ok())
    {
        vector<featurept> featurepts;
        ImageConverter::find_danger_points(ic.img_1,featurepts);
        //cacualate_pre_integreation();
        ros::spinOnce();
        loop_rate.sleep();

    }

}
