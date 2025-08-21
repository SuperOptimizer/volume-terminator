// main.cpp
// Chao Du 2014 Dec

#include <qapplication.h>

#include "CWindow.hpp"

#include <thread>



auto main(int argc, char* argv[]) -> int
{
    cv::setNumThreads(std::thread::hardware_concurrency());
    
    QApplication app(argc, argv);
    QApplication::setOrganizationName("Vesuvius Challenge Team");
    QApplication::setApplicationName("VT");
    QApplication::setWindowIcon(QIcon(":/images/logo.png"));
    QApplication::setApplicationVersion(QString::fromStdString("1.0"));

    CWindow aWin;
    aWin.show();
    return QApplication::exec();
}
