#ifndef FACE_LEARN_H
#define FACE_LEARN_H

#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

class FaceLearn {
private:
    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;
    image_transport::Publisher image_pub;

    Ptr<cuda::CascadeClassifier> face;
    Ptr<face::FaceRecognizer> recognize;

    bool is_model_trained_;
    int next_label_;
    double running_average_;
    vector<cv::Mat> sample_images_;
    vector<int> sample_lablels_;
public:
    FaceLearn(ros::NodeHandle nh);
    ~FaceLearn();
    void detectFaces(const sensor_msgs::ImageConstPtr &msg);
    void recognizeFaces(const cv_bridge::CvImagePtr cv_ptr, vector<cv::Rect> faces);
};

#endif //FACE_LEARN_H

