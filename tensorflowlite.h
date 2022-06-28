#ifndef TENSORFLOWLITE_H
#define TENSORFLOWLITE_H


#include <iostream>
/*** Include ***/
/* for general */
#include <stdint.h>
#include <stdio.h>
//#include <fstream>
#include <vector>
#include <string>
#include <chrono>


#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"


/* for Edge TPU */
#include "include/edgetpu.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor.h"

#include "include/model_utils.h"

#include <QString>
#include <QXmlStreamWriter>
#include <QFile>
#include <QDir>
#include <QJsonObject>
#include <QByteArray>
#include <QJsonDocument>
#include <QDebug>

#define NUM_MAX_RESULT 100

typedef struct {
    char     workDir[256];
    int32_t  numThreads;
} INPUT_PARAM;

typedef struct {
    int32_t  classId;
    char     label[256];
    double score;
    int32_t  x;
    int32_t  y;
    int32_t  width;
    int32_t  height;
    bool hasCrop;
    cv::Mat  crop;
    char     cropLabel[256];
    char     holeLabel[256];
} BBOX;

typedef struct {
    int32_t objectNum;
    std::vector<BBOX> objectList;
    double timePreProcess;   // [msec]
    double timeInference;    // [msec]
    double timePostProcess;  // [msec]
} OUTPUT_OBJ;

typedef struct {
    int32_t  classId;
    char     label[256];
    double score;
    cv::Mat  crop;
    double timePreProcess;   // [msec]
    double timeInference;    // [msec]
    double timePostProcess;  // [msec]
} OUTPUT_CLASS;

/* Model parameters */
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224

//#define MODEL_WIDTH_DET 300
#define MODEL_WIDTH_DET 400

#define MODEL_HEIGHT_DET 300

//#define MODEL_WIDTH_DET 320
//#define MODEL_HEIGHT_DET 320

#define MODEL_CHANNEL 3
#define NUM_MAX_RESULT 100

class Tensorflowlite
{
public:

    Tensorflowlite(int tpu_num, const char *model_f, const char *model_l, float threshold, bool tpu);
    ~Tensorflowlite();

    void readLabel();

    void runClass();
    void setFrame(const cv::Mat &image);
    void runDet();
    OUTPUT_OBJ getObjResults() const;

    std::vector<std::string> getLabels() const;

    cv::Rect cropMat(int h, int w, int xmin, int ymin, int xmax, int ymax);
    OUTPUT_CLASS getOutClass() const;

    void xmlSet(int w, int h, int xmin, int ymin, int, int ymax, QString result, QString dir, QString time);

    std::string getJson(int w, int h, int xmin, int ymin, int xmax, int ymax, QString detectClass, QString dirLabel, QString time,
                        QString cropName, QString cropClassChild);
    void runAnom();
    void runFace();
    void runLand();
    void runFt();
private:
    const char* model_f;
    const char* model_l;
    std::vector<std::string> labels;
    cv::Mat frame; // percept itself

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::vector<TfLiteTensor*> outputs;


    OUTPUT_OBJ outResults;
    OUTPUT_CLASS outClass;

    float threshold;

    /*** Global variable ***/
    //std::unique_ptr<DetectionEngine> s_engine;
};


#endif // TENSORFLOWLITE_H
