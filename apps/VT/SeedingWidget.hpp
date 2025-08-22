#pragma once

#include <QWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QHBoxLayout>
#include <QProcess>
#include <QPointer>
#include <opencv2/core.hpp>
#include <memory>

#include "CSurfaceCollection.hpp"
#include "PathData.hpp"
#include "VCCollection.hpp"
#include "VolumePkg.hpp"

class SeedingWidget : public QWidget {
    Q_OBJECT
    
public:
    explicit SeedingWidget(VCCollection* point_collection, CSurfaceCollection* surface_collection, QWidget* parent = nullptr);
    ~SeedingWidget() override;
    
    void setVolumePkg(const std::shared_ptr<VolumePkg> &vpkg);
    void setCurrentVolume(const std::shared_ptr<Volume> &volume);
    void setCache(ChunkCache* cache);
    
signals:
    void sendPathsChanged(const QList<PathData>& paths);
    void sendStatusMessageAvailable(QString text, int timeout);
    
public slots:
    void onSurfacesLoaded() const;  // Called when surfaces have been loaded/reloaded
    void onCollectionAdded(uint64_t collectionId) const;
    void onCollectionChanged(uint64_t collectionId) const;
    void onCollectionRemoved(uint64_t collectionId) const;
    
public slots:
    void onVolumeChanged(const std::shared_ptr<Volume> &vol, const std::string& volumeId);
    void updateCurrentZSlice(int z);
    void onMousePress(const cv::Vec3f &vol_point, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(const cv::Vec3f &vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(cv::Vec3f vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    
private slots:
    void onPreviewRaysClicked();
    void onClearPreviewClicked() const;
    void onCastRaysClicked();
    void onClearPeaksClicked() const;
    void onRunSegmentationClicked();
    void onExpandSeedsClicked();
    void onResetPointsClicked();
    void onCancelClicked();
    
private:
    // Mode enum
    enum class Mode {
        PointMode,
        DrawMode
    };
    
    void setupUI();
    void computeDistanceTransform();
    void castRays();
    void findPeaksAlongRay(const cv::Vec2f& rayDir, const cv::Vec3f& startPoint) const;

    static QString findExecutablePath();
    void updateParameterPreview() const;
    void updateModeUI() const;
    void analyzePaths();
    void findPeaksAlongPath(const PathData& path) const;
    void startDrawing(const cv::Vec3f &startPoint);
    void addPointToPath(const cv::Vec3f &point);
    void finalizePath();
    QColor generatePathColor();
    void displayPaths();

    static void updatePointsDisplay();
    void updateInfoLabel() const;
    void updateButtonStates() const;
    
private:
    
    // UI elements
    QLabel* infoLabel;
    QComboBox* collectionComboBox;
    QDoubleSpinBox* angleStepSpinBox;
    QSpinBox* processesSpinBox;
    QSpinBox* thresholdSpinBox;  // Intensity threshold for peak detection
    QSpinBox* windowSizeSpinBox; // Window size for peak detection
    QSpinBox* maxRadiusSpinBox;  // Max radius for ray casting
    QSpinBox* expansionIterationsSpinBox; // Number of expansion iterations
    
    // Layout and label references for hiding/showing
    QHBoxLayout* maxRadiusLayout;
    QLabel* maxRadiusLabel;
    QHBoxLayout* angleStepLayout;
    QLabel* angleStepLabel;
    
    QString executablePath;
    
    QPushButton* previewRaysButton;
    QPushButton* clearPreviewButton;
    QPushButton* castRaysButton;
    QPushButton* clearPeaksButton;
    QPushButton* runSegmentationButton;
    QPushButton* expandSeedsButton;
    QPushButton* resetPointsButton;
    QPushButton* cancelButton;
    QProgressBar* progressBar;
    
    // Data
    std::shared_ptr<VolumePkg> fVpkg;
    std::shared_ptr<Volume> currentVolume;
    std::string currentVolumeId;
    ChunkCache* chunkCache;
    int currentZSlice;
    VCCollection* _point_collection;
    CSurfaceCollection* _surface_collection;
    cv::Mat distanceTransform;
    
    // Drawing mode data
    Mode currentMode;
    QList<PathData> paths;  
    bool isDrawing;
    PathData currentPath;
    int colorIndex;
    
    // Process management
    QList<QPointer<QProcess>> runningProcesses;
    bool jobsRunning;
};

