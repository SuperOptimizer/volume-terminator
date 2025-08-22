#pragma once

#include <qimage.h>
#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// Convert from cv::Mat to QImage
static inline QImage Mat2QImage(const cv::Mat& nSrc)
{
    cv::Mat tmp;
    cvtColor(nSrc, tmp, cv::COLOR_BGR2RGB);  // copy and convert color space
    QImage result(
        static_cast<const std::uint8_t*>(tmp.data), tmp.cols, tmp.rows,
        tmp.step, QImage::Format_RGB888);
    result.bits();  // enforce depp copy, see documentation of
    // QImage::QImage( const uchar *dta, int width, int height, Format format )
    return result;
}


