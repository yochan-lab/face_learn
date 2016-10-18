#include "face_learn.h"

//using namespace cv::cuda;

FaceLearn::FaceLearn(ros::NodeHandle nh):it(nh) {
    std::string cascade_location, camera_stream, face_database;

    // Retrieve the location of the trained haar cascade from the parameter server
    nh.getParam("cascade_file", cascade_location);
    face = cuda::CascadeClassifier::create(cascade_location);               // load the GPU haar cascade
    ROS_INFO("Using cascade from: %s", cascade_location.c_str());

    // Retrieve the topic of the image from the parameter server
    nh.getParam("camera_stream", camera_stream);
    image_sub = it.subscribe(camera_stream, 1, &FaceLearn::detectFaces, this); // subscribe to the camera feed
    image_pub = it.advertise("/hout", 1);
    // TODO: catch exception
    ROS_INFO("Using topic: %s", camera_stream.c_str());

    // Retrieve the location of the face database from the parameter server
    nh.getParam("face_database_file", face_database);
    recognize = cv::face::createLBPHFaceRecognizer();
    next_label_ = rand();
    try {
        recognize->load(face_database);
        is_model_trained_ = true;
    } catch (Exception &e) {
        // is untrained
        is_model_trained_ = false;
        ROS_ERROR("A new database will be created at %s", face_database.c_str());
    }

    ROS_INFO("waiting for images");
}

FaceLearn::~FaceLearn() {}

void FaceLearn::detectFaces(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;
    vector<cv::Rect> faces;
 
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge_exception: %s", e.what());
        return;
    }

    frame_gpu.upload(cv_ptr->image);

    // face->setFindLargestObject(true);
    face->setScaleFactor(1.2);
    face->setMinNeighbors(6);
    face->detectMultiScale(frame_gpu, facesBuf_gpu);
    face->convert(facesBuf_gpu, faces);

    if (!faces.empty()) {
        // do stuff here!
        // ROS_INFO("FOUND FACE!");
        recognizeFaces(cv_ptr, faces);
    } else {
        // ROS_INFO("NO FACE FOUND");
    }
}

void FaceLearn::recognizeFaces(cv_bridge::CvImagePtr cv_ptr, vector<cv::Rect> faces) {
    int result_label;
    double result_confidence;
    cv_bridge::CvImagePtr labeled_image;
    
    // label image
    //labeled_image->image = cv_ptr->image;
    vector<cv::Rect>::iterator it;
    ROS_INFO("%d", faces.size());
    for(it = faces.begin(); it != faces.end(); it++) {
        rectangle(cv_ptr->image, *it, Scalar(255,0,0));
    }
    //cv_ptr->encoding = sensor_msgs::image_encodings::MONO8;
    image_pub.publish(cv_ptr->toImageMsg());
    
    // crop the face
    cv_ptr->image = cv_ptr->image(faces[0]).clone();
    cv_ptr->encoding = sensor_msgs::image_encodings::MONO8;
    //image_pub.publish(cv_ptr->toImageMsg());

    sample_images_.push_back(cv_ptr->image);
    sample_lablels_.push_back(next_label_);

    // the model is not yet trained, empty file?
    // create a new data point.
    // this should run at most once
    if (!is_model_trained_ && sample_images_.size() >10) {
        // train ... with a sample of 10
        recognize->train(sample_images_, sample_lablels_);
        is_model_trained_ = true;
        next_label_ = rand();

        // remove training data
        sample_images_.erase(sample_images_.begin(), sample_images_.end());
        sample_lablels_.erase(sample_lablels_.begin(), sample_lablels_.end());
        return;
    }

    // recognition step
    if (is_model_trained_) {
        // ROS_INFO("processing face...");
        recognize->predict(cv_ptr->image, result_label, result_confidence);
        running_average_ = (0.8)*result_confidence + (0.2)*running_average_;
        ROS_INFO("Result: I saw %i: %f", result_label, running_average_);
    }

    // update step
    if (sample_images_.size() > 10) {
    // if we meet somebody new, this number will be high
        if(running_average_ > 100) {
            recognize->update(sample_images_, sample_lablels_);
            ROS_INFO("NEW FACE");
            next_label_ = rand();
        }
        // reset for the next face
        sample_images_.erase(sample_images_.begin(), sample_images_.end());
        sample_lablels_.erase(sample_lablels_.begin(), sample_lablels_.end());
        running_average_ = 0;
    }
}