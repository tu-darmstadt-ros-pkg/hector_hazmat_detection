#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ocl/ocl.hpp"

#include <vector>

#include <dirent.h>
#include <fstream>

#include "hector_hazmat_detection/LinearSVM.h"

//#include <libsvm/svm.h>

using namespace cv;
using namespace std;

// Directory containing positive sample images
static string posSamplesDir = "imgs/positives/";
// Directory containing negative sample images
static string negSamplesDir = "imgs/negatives/";
// Set the file to write the features to
static string featuresFile = "genfiles/features.dat";
// Set the file to write the SVM model to
static string svmModelFile = "genfiles/svm.yaml";
// Set the file to write the resulting detecting descriptor vector to
static string descriptorVectorFile = "genfiles/descriptorvector.dat";
// Set the file to write the resulting opencv hog classifier as YAML file
static string cvHOGFile = "genfiles/hog.yaml";

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);

static const Size win_size = Size(64, 64);

Mat get_hogdescriptor_visual_image(Mat& origImg,
  vector<float>& descriptorValues,
  Size winSize,
  Size cellSize,
  int scaleFactor,
  double viz_factor);

  static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions);

  static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog);

  static void detectTrainingSetTest(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames);

  static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData);
  static void showDetections(const vector<Point>& found, Mat& imageData);
  static void showDetections(const vector<Rect>& found, Mat& imageData);

  static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
      t += tolower(*i);
    }
    return t;
  }

  int main(int argc, char **argv)
  {
    ros::Time::init();

    // <editor-fold defaultstate="collapsed" desc="Init">
    HOGDescriptor hog; // Use standard parameters here
    hog.winSize = win_size; // Default training images size as used in paper
    // Train the SVM
    LinearSVM svm;

    svm.load(svmModelFile.c_str());

    vector<float> support_vector;
    svm.getSupportVector(support_vector);

    hog.setSVMDetector(support_vector);

    hog.load(cvHOGFile);


    ROS_INFO("Hog Detector import from: %s", cvHOGFile.c_str());
    double hitThreshold = 0; //Just ommit this

    Mat testImage = imread("imgs/hazmat-test.jpg");
    cvtColor(testImage, testImage, CV_RGB2GRAY);

    clock_t start = clock();
    detectTest(hog, hitThreshold, testImage);
    clock_t ticks = clock()-start;
    ROS_INFO("Detection time: %f", (double)ticks/CLOCKS_PER_SEC);

    imshow("HOG custom detection", testImage);
    waitKey(0);


    return EXIT_SUCCESS;

  }

  Mat get_hogdescriptor_visual_image(Mat& origImg,
    vector<float>& descriptorValues,
    Size winSize,
    Size cellSize,
    int scaleFactor,
    double viz_factor)
    {
      Mat visual_image;
      resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));

      int gradientBinSize = 9;
      // dividing 180° into 9 bins, how large (in rad) is one bin?
      float radRangeForOneBin = 3.14/(float)gradientBinSize;

      // prepare data structure: 9 orientation / gradient strenghts for each cell
      int cells_in_x_dir = winSize.width / cellSize.width;
      int cells_in_y_dir = winSize.height / cellSize.height;
      int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
      float*** gradientStrengths = new float**[cells_in_y_dir];
      int** cellUpdateCounter   = new int*[cells_in_y_dir];
      for (int y=0; y<cells_in_y_dir; y++)
      {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
          gradientStrengths[y][x] = new float[gradientBinSize];
          cellUpdateCounter[y][x] = 0;

          for (int bin=0; bin<gradientBinSize; bin++)
          gradientStrengths[y][x][bin] = 0.0;
        }
      }

      // nr of blocks = nr of cells - 1
      // since there is a new block on each cell (overlapping blocks!) but the last one
      int blocks_in_x_dir = cells_in_x_dir - 1;
      int blocks_in_y_dir = cells_in_y_dir - 1;

      // compute gradient strengths per cell
      int descriptorDataIdx = 0;
      int cellx = 0;
      int celly = 0;

      for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
      {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
          // 4 cells per block ...
          for (int cellNr=0; cellNr<4; cellNr++)
          {
            // compute corresponding cell nr
            int cellx = blockx;
            int celly = blocky;
            if (cellNr==1) celly++;
            if (cellNr==2) cellx++;
            if (cellNr==3)
            {
              cellx++;
              celly++;
            }

            for (int bin=0; bin<gradientBinSize; bin++)
            {
              float gradientStrength = descriptorValues[ descriptorDataIdx ];
              descriptorDataIdx++;

              gradientStrengths[celly][cellx][bin] += gradientStrength;

            } // for (all bins)


            // note: overlapping blocks lead to multiple updates of this sum!
            // we therefore keep track how often a cell was updated,
            // to compute average gradient strengths
            cellUpdateCounter[celly][cellx]++;

          } // for (all cells)


        } // for (all block x pos)
      } // for (all block y pos)


      // compute average gradient strengths
      for (int celly=0; celly<cells_in_y_dir; celly++)
      {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {

          float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

          // compute average gradient strenghts for each gradient bin direction
          for (int bin=0; bin<gradientBinSize; bin++)
          {
            gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
          }
        }
      }


      cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

      // draw cells
      for (int celly=0; celly<cells_in_y_dir; celly++)
      {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
          int drawX = cellx * cellSize.width;
          int drawY = celly * cellSize.height;

          int mx = drawX + cellSize.width/2;
          int my = drawY + cellSize.height/2;

          rectangle(visual_image,
            Point(drawX*scaleFactor,drawY*scaleFactor),
            Point((drawX+cellSize.width)*scaleFactor,
            (drawY+cellSize.height)*scaleFactor),
            CV_RGB(100,100,100),
            1);

            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
              float currentGradStrength = gradientStrengths[celly][cellx][bin];

              // no line to draw?
              if (currentGradStrength==0)
              continue;

              float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

              float dirVecX = cos( currRad );
              float dirVecY = sin( currRad );
              float maxVecLen = cellSize.width/2;
              float scale = viz_factor; // just a visual_imagealization scale,
              // to see the lines better

              // compute line coordinates
              float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
              float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
              float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
              float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

              // draw gradient visual_imagealization
              line(visual_image,
                Point(x1*scaleFactor,y1*scaleFactor),
                Point(x2*scaleFactor,y2*scaleFactor),
                CV_RGB(0,0,255),
                1);

              } // for (all bins)

            } // for (cellx)
          } // for (celly)


          // don't forget to free memory allocated by helper data structures!
          for (int y=0; y<cells_in_y_dir; y++)
          {
            for (int x=0; x<cells_in_x_dir; x++)
            {
              delete[] gradientStrengths[y][x];
            }
            delete[] gradientStrengths[y];
            delete[] cellUpdateCounter[y];
          }
          delete[] gradientStrengths;
          delete[] cellUpdateCounter;

          return visual_image;

        }

        /**
        * For unixoid systems only: Lists all files in a given directory and returns a vector of path+name in string format
        * @param dirName
        * @param fileNames found file names in specified directory
        * @param validExtensions containing the valid file extensions for collection in lower case
        */
        static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
          struct dirent* ep;
          size_t extensionLocation;
          DIR* dp = opendir(dirName.c_str());
          if (dp != NULL) {
            while ((ep = readdir(dp))) {

              extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
              // Check if extension is matching the wanted ones
              string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
              if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                ROS_DEBUG("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
              } else {
                ROS_DEBUG("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
              }
            }
            (void) closedir(dp);
          } else {
            ROS_ERROR("Error opening directory '%s'!\n", dirName.c_str());
          }
          return;
        }


        /**
        * This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
        * @param imageFilename file path of the image file to read and calculate feature vector from
        * @param descriptorVector the returned calculated feature vector<float> ,
        *      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. Mat<float>)
        * @param hog HOGDescriptor containin HOG settings
        */
        static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
          /** for imread flags from openCV documentation,
          * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
          * @note If you get a compile-time error complaining about following line (esp. imread),
          * you either do not have a current openCV version (>2.0)
          * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
          */
          ROS_DEBUG("Reading file: %s", imageFilename.c_str());
          Mat imageData = imread(imageFilename);
          ROS_DEBUG("Cols: %d, Rows: %d", imageData.cols, imageData.rows);

          //Mat imageData = imread(imageFilename, IMREAD_GRAYSCALE);
          resize(imageData, imageData, win_size );
          cvtColor(imageData, imageData, CV_RGB2GRAY);
          if (imageData.empty()) {
            featureVector.clear();
            ROS_WARN("HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
            return;
          }
          // Check for mismatching dimensions
          if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
            featureVector.clear();
            ROS_ERROR("Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
            return;
          }
          vector<Point> locations;
          hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
          imageData.release(); // Release the image again after features are extracted
        }


        /**
        * Test the trained detector against the same training set to get an approximate idea of the detector.
        * Warning: This does not allow any statement about detection quality, as the detector might be overfitting.
        * Detector quality must be determined using an independent test set.
        * @param hog
        */
        static void detectTrainingSetTest(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames) {
          unsigned int truePositives = 0;
          unsigned int trueNegatives = 0;
          unsigned int falsePositives = 0;
          unsigned int falseNegatives = 0;
          vector<Point> foundDetection;
          // Walk over positive training samples, generate images and detect
          for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
            const Mat imageData = imread(*posTrainingIterator, IMREAD_GRAYSCALE);
            hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
            ROS_INFO("Processing positive: %s", posTrainingIterator->c_str());
            ROS_INFO("Detections: %d", foundDetection.size());
            if (foundDetection.size() > 0) {
              ++truePositives;
              falseNegatives += foundDetection.size() - 1;
            } else {
              ++falseNegatives;
            }
          }
          // Walk over negative training samples, generate images and detect
          for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
            const Mat imageData = imread(*negTrainingIterator, IMREAD_GRAYSCALE);
            hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
            ROS_INFO("Processing negative: %s", negTrainingIterator->c_str());
            ROS_INFO("Detections: %d", foundDetection.size());
            if (foundDetection.size() > 0) {
              falsePositives += foundDetection.size();
            } else {
              ++trueNegatives;
            }
          }

          ROS_INFO("Results:");
          ROS_INFO("True Positives: %u", truePositives);
          ROS_INFO("True Negatives: %u", trueNegatives);
          ROS_INFO("False Positives: %u", falsePositives);
          ROS_INFO("False Negatives: %u", falseNegatives);

        }

        static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData) {
          vector<Rect> found;
          Size padding(Size(8, 8));
          Size winStride(Size(8, 8));
          hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding);
          showDetections(found, imageData);
        }


        /**
        * Shows the detections in the image
        * @param found vector containing valid detection rectangles
        * @param imageData the image in which the detections are drawn
        */
        static void showDetections(const vector<Point>& found, Mat& imageData) {
          size_t i, j;
          for (i = 0; i < found.size(); ++i) {
            Point r = found[i];
            // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
            rectangle(imageData, Rect(r.x-16, r.y-32, 32, 64), Scalar(64, 255, 64), 3);
          }
        }

        /**
        * Shows the detections in the image
        * @param found vector containing valid detection rectangles
        * @param imageData the image in which the detections are drawn
        */
        static void showDetections(const vector<Rect>& found, Mat& imageData) {
          vector<Rect> found_filtered;
          size_t i, j;
          for (i = 0; i < found.size(); ++i) {
            Rect r = found[i];
            for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
            break;
            if (j == found.size())
            found_filtered.push_back(r);
          }
          for (i = 0; i < found_filtered.size(); i++) {
            Rect r = found_filtered[i];
            rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
          }
        }
