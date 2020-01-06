#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;
// the path of file
string left_file = "../KITTI_Left.png";
string right_file = "../KITTI_Right.png";
// ploting in the pangolin
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv) {

    // intristic parameter
      double fx = 732.5377, fy = 721.5377, cx = 609.5593, cy = 172.8540;
    // base line, maybe there is something wring in the pdf, it is −3.875744e + 02???? 381.57 meters????
      double b =0.3875744;

    // imread image
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);
    //  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
    //      0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);  //magic digitals
    //we can also use the StereoBM nethod, but it didn't perform perfectly
    cv::Ptr<cv::StereoBM> sgbm = cv::StereoBM::create(64,17);
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    // get the pointscloud
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    // for loop
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // the first three dimension are the coordinates in the space, the last dimension is the color dimension

            // get the position of point
            double x = (u - cx) / fx;//normalized coordinates
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));// z = fb/d  here b is distance of camrea, d = ul - ur
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;//true 3D points in the space

            pointcloud.push_back(point);
        }
    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    // ploting pointscloud
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
