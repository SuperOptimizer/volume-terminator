#pragma once

#include <QWidget>
#include <QList>
#include <QMap>
#include <QColor>
#include <memory>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QSlider>
#include <QSpinBox>
#include <opencv2/core.hpp>

#include "PathData.hpp"
#include "Slicing.hpp"
#include "VolumePkg.hpp"

class DrawingWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit DrawingWidget(QWidget* parent = nullptr);
    ~DrawingWidget() override;
    
    /** Set the volume package */
    void setVolumePkg(const std::shared_ptr<VolumePkg> &vpkg);
    
    /** Set the current volume */
    void setCurrentVolume(const std::shared_ptr<Volume> &volume);
    
    /** Set the cache for volume data access */
    void setCache(ChunkCache* cache);
    
    /** Clear all drawn paths */
    void clearAllPaths();
    
    /** Get current path ID */
    int getCurrentPathId() const { return currentPathId; }
    
    /** Get current brush size */
    float getBrushSize() const { return brushSize; }
    
    /** Check if in eraser mode */
    bool isEraserMode() const { return eraserMode; }
    
    /** Check if drawing mode is active */
    bool isDrawingModeActive() const { return drawingModeActive; }
    
    /** Get current brush shape */
    PathData::BrushShape getBrushShape() const { return brushShape; }
    
    /** Toggle drawing mode */
    void toggleDrawingMode() const;

public slots:
    /** Handle volume change */
    void onVolumeChanged(const std::shared_ptr<Volume>& vol);
    void onVolumeChanged(const std::shared_ptr<Volume> &vol, const std::string& volumeId);
    
    /** Handle mouse events from volume viewers */
    void onMousePress(const cv::Vec3f& vol_point, const cv::Vec3f& normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(const cv::Vec3f &vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(const cv::Vec3f& vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    
    /** Handle Z-slice changes */
    void updateCurrentZSlice(int z);
    
    /** Handle surface reload */
    void onSurfacesLoaded() const;

signals:
    /** Emitted when paths change */
    void sendPathsChanged(const QList<PathData>& paths);
    
    /** Emitted to show status messages */
    void sendStatusMessageAvailable(const QString& message, int timeout);
    
    /** Emitted when drawing mode is active/inactive */
    void sendDrawingModeActive(bool active);

private slots:
    /** UI control handlers */
    void onPathIdChanged(int value);
    void onBrushSizeChanged(int value);
    void onOpacityChanged(int value);
    void onEraserToggled(bool checked);
    void onBrushShapeChanged();
    void onClearAllClicked();
    void onSaveAsMaskClicked();
    void onColorButtonClicked();

private:
    /** Initialize UI components */
    void setupUI();
    
    /** Update UI based on current state */
    void updateUI() const;
    
    /** Start drawing a new path */
    void startDrawing(const cv::Vec3f& startPoint);
    
    /** Add point to current path */
    void addPointToPath(const cv::Vec3f &point);
    
    /** Finalize current path */
    void finalizePath();
    
    /** Get or create color for path ID */
    QColor getColorForPathId(int pathId);
    
    /** Update the color preview button */
    void updateColorPreview();
    
    /** Generate mask from drawn paths */
    cv::Mat generateMask() const;
    
    /** Save mask to file */
    static void saveMask(const cv::Mat& mask, const std::string& filename);
    
    /** Check if a volume point is valid (within bounds and not -1) */
    bool isValidVolumePoint(const cv::Vec3f& point) const;
    
    /** Process paths to apply eraser operations */
    static QList<PathData> processPathsWithErasers(const QList<PathData>& rawPaths);
    
    /** Calculate distance from point to line segment */
    static float pointToSegmentDistance(const cv::Vec3f& point, const cv::Vec3f& segStart, const cv::Vec3f& segEnd);
    
    /** Check if a point is within eraser brush */
    static bool isPointInEraserBrush(const cv::Vec3f& point, const cv::Vec3f& eraserPoint,
                                     float eraserRadius, PathData::BrushShape brushShape);

private:
    // Volume data
    std::shared_ptr<VolumePkg> fVpkg;
    std::shared_ptr<Volume> currentVolume;
    std::string currentVolumeId;
    ChunkCache* chunkCache;
    
    // Drawing state
    int currentPathId;
    float brushSize;
    float opacity;
    bool eraserMode;
    PathData::BrushShape brushShape;
    
    // Path management
    QList<PathData> drawnPaths;
    PathData currentPath;
    bool isDrawing;
    cv::Vec3f lastPoint;
    int currentZSlice;
    
    // Color management
    QMap<int, QColor> pathIdColors;
    
    // Drawing mode state
    bool drawingModeActive;
    
    // Temporary eraser state (for shift-to-erase)
    bool temporaryEraserMode;
    bool originalEraserMode;
    
    // UI elements
    QLabel* infoLabel;
    QPushButton* toggleModeButton;
    QSpinBox* pathIdSpinBox;
    QSlider* brushSizeSlider;
    QLabel* brushSizeLabel;
    QSlider* opacitySlider;
    QLabel* opacityLabel;
    QCheckBox* eraserCheckBox;
    QButtonGroup* brushShapeGroup;
    QRadioButton* circleRadio;
    QRadioButton* squareRadio;
    QPushButton* colorButton;
    QPushButton* clearAllButton;
    QPushButton* saveAsMaskButton;
};

