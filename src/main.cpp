// #include <iostream>
// #include <iomanip>
// #include "opencv2/opencv.hpp"
// #include "opencv2/objdetect/objdetect.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/cudaobjdetect.hpp"
// #include "opencv2/cudaimgproc.hpp"
// #include "opencv2/cudawarping.hpp"

#include "ros/ros.h"
// #include "std_msgs/String.h"

// #include "image_transport/image_transport.h"
// #include "cv_bridge/cv_bridge.h"
// #include "sensor_msgs/image_encodings.h"

#include "face_learn.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main (int argc, char *argv[]) {
    ros::init(argc, argv, "face_learn");
    ros::NodeHandle n("~");

    if (!n.hasParam("cascade_file")) {
        ROS_ERROR("NO 'cascade_file' parameter specified. exiting.");
    }

    if (!n.hasParam("face_database_file")) {
        ROS_ERROR("No 'face_database_file' parameter specified. exiting.");
        return 0;
    }

    if (!n.hasParam("camera_stream")) {
        ROS_ERROR("No 'camera_stream' parameter specified. exiting.");
        return 0;
    }

    if(getCudaEnabledDeviceCount() == 0) {
        ROS_ERROR("No GPU found!");
        return 0;
    }

    FaceLearn d(n);
    ros::spin();

    //cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    return 0;
} 