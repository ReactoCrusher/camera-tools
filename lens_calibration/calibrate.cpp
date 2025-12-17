#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <numeric>
#include <iomanip>

#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "cxxopts.hpp"
#include "cnpy.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

static vector<Mat> read_images(const string &directory)
{
    vector<Mat> images;
    for (const auto &entry : fs::directory_iterator(directory))
    {
        Mat img = imread(entry.path().string());
        if (!img.empty())
        {
            images.push_back(img);
        }
    }
    return images;
}

static void calibrate_camera_from_images(const vector<Mat> &images, Size board_size, float square_size, const std::string& output_file)
{
    vector<vector<Point2f>> image_points;
    vector<vector<Point3f>> object_points;
    vector<Point3f> obj;

    for (int i = 0; i < board_size.height; ++i)
    {
        for (int j = 0; j < board_size.width; ++j)
        {
            obj.push_back(Point3f(j * square_size, i * square_size, 0));
        }
    }

    for (const auto &image : images) {
        vector<Point2f> corners;
        bool found = findChessboardCorners(image, board_size, corners);
        if (found)
        {
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));
            Mat canvas = image.clone();
            drawChessboardCorners(canvas, board_size, corners, found);
            imshow("Chessboard", canvas);
            waitKey(10);
            image_points.push_back(corners);
            object_points.push_back(obj);
        }
    }

    std::cout << "Found " << image_points.size() << " images" << std::endl;

    if (image_points.empty())
    {
        std::cerr << "No valid chessboard detections found. Aborting calibration." << std::endl;
        return;
    }

    Mat camera_matrix = Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = 1500.0;
    camera_matrix.at<double>(1, 1) = 1500.0;
    camera_matrix.at<double>(0, 2) = images[0].size().width / 2.0;
    camera_matrix.at<double>(1, 2) = images[0].size().height / 2.0;

    Mat dist_coeffs = Mat::zeros(1, 5, CV_64F);

    vector<Mat> rvecs, tvecs;
    const double rms = calibrateCamera(
        object_points,
        image_points,
        images[0].size(),
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        CALIB_USE_INTRINSIC_GUESS | CALIB_FIX_ASPECT_RATIO | 
        CALIB_FIX_PRINCIPAL_POINT | CALIB_ZERO_TANGENT_DIST |
         CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | 
         CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6
    );

    std::cout << "Camera matrix: " << camera_matrix << std::endl;
    std::cout << "Distortion coefficients: " << dist_coeffs << std::endl;

    // Compute per-view and overall reprojection RMSE (in pixels)
    vector<double> per_view_rmse;
    per_view_rmse.reserve(object_points.size());
    double total_squared_error = 0.0;
    size_t total_points = 0;

    for (size_t i = 0; i < object_points.size(); ++i)
    {
        vector<Point2f> projected;
        projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs, projected);

        double view_squared_error = 0.0;
        for (size_t j = 0; j < projected.size(); ++j)
        {
            const double dx = static_cast<double>(image_points[i][j].x) - static_cast<double>(projected[j].x);
            const double dy = static_cast<double>(image_points[i][j].y) - static_cast<double>(projected[j].y);
            view_squared_error += dx * dx + dy * dy;
        }

        const size_t num_points = projected.size();
        const double view_rmse = std::sqrt(view_squared_error / static_cast<double>(num_points));
        per_view_rmse.push_back(view_rmse);
        total_squared_error += view_squared_error;
        total_points += num_points;
    }

    const double overall_rmse = std::sqrt(total_squared_error / static_cast<double>(total_points));
    const double mean_rmse = std::accumulate(per_view_rmse.begin(), per_view_rmse.end(), 0.0) / static_cast<double>(per_view_rmse.size());
    double variance_rmse = 0.0;
    for (const double e : per_view_rmse)
    {
        const double d = e - mean_rmse;
        variance_rmse += d * d;
    }
    variance_rmse /= static_cast<double>(per_view_rmse.size());
    const double stddev_rmse = std::sqrt(variance_rmse);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Calibration RMS (OpenCV): " << rms << " px" << std::endl;
    std::cout << "Overall reprojection RMSE: " << overall_rmse << " px" << std::endl;
    std::cout << "Per-image RMSE: mean " << mean_rmse << " px, stddev " << stddev_rmse << " px" << std::endl;
    std::cout.unsetf(std::ios_base::floatfield);

    cnpy::npz_save(output_file, "camMatrix", (double*)camera_matrix.data, {3, 3}, "w");
    cnpy::npz_save(output_file, "distCoef", (double*)dist_coeffs.data, {(size_t)dist_coeffs.rows, (size_t)dist_coeffs.cols}, "a");
    // Save quality metrics for later analysis
    cnpy::npz_save(output_file, "rms", &rms, {1}, "a");
    cnpy::npz_save(output_file, "overall_rmse", &overall_rmse, {1}, "a");
    cnpy::npz_save(output_file, "perViewErrors", per_view_rmse.data(), {per_view_rmse.size()}, "a");

    for (const auto &image : images)
    {
        Mat undistorted_image;
        undistort(image, undistorted_image, camera_matrix, dist_coeffs);
        imshow("Undistorted Image", undistorted_image);
        waitKey(50);
    }
}

int main(int argc, char** argv)
{
    cxxopts::Options options("Calibrate");
    options.add_options()
        ("d,directory", "Directory containing images", cxxopts::value<string>())
        ("w,width", "Width of the chessboard pattern", cxxopts::value<int>())
        ("h,height", "Height of the chessboard pattern", cxxopts::value<int>())
        ("s,square", "Size of the chessboard square", cxxopts::value<float>())
        ("o,output", "Output file name", cxxopts::value<string>())
        ("help", "Print usage");

    const auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        cout << options.help() << endl;
        return 0;
    }

    const string directory = result["directory"].as<string>();
    const int board_width = result["width"].as<int>();
    const int board_height = result["height"].as<int>();
    const float square_size = result["square"].as<float>();
    const string output_file = result["output"].as<string>();

    const Size board_size(board_width, board_height);
    const auto images = read_images(directory);

    if (images.empty())
    {
        cout << "No images found in the directory." << endl;
        return -1;
    }

    calibrate_camera_from_images(images, board_size, square_size, output_file);

    return 0;
}
