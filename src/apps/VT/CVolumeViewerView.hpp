#pragma once
#include <QGraphicsView>



class CVolumeViewerView : public QGraphicsView
{
    Q_OBJECT
    
public:
    CVolumeViewerView(QWidget* parent = nullptr);
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void scrollContentsBy(int dx, int dy) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    /// Set physical voxel size (units per scene-unit, e.g. µm/pixel).
    /// Call this after you load your Zarr spacing metadata.
    void setVoxelSize(double sx, double sy) { m_vx = sx; m_vy = sy; update(); }

signals:
    void sendResized();
    void sendScrolled();
    void sendZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers);
    void sendVolumeClicked(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanRelease(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendPanStart(Qt::MouseButton, Qt::KeyboardModifiers);
    void sendCursorMove(QPointF);
    void sendMousePress(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendMouseMove(QPointF, Qt::MouseButtons, Qt::KeyboardModifiers);
    void sendMouseRelease(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    void sendKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    
protected:
    bool _regular_pan = false;
    QPoint _last_pan_position;
    bool _left_button_pressed = false;
    /// Draw our scalebar on every repaint
    void drawForeground(QPainter* painter, const QRectF& sceneRect) override;

 private:
    /// Round “ideal” length to 1,2 or 5 × 10^n
    static double chooseNiceLength(double nominal);

    // µm per scene-unit (pixel)  
    double m_vx = 32.0, m_vy = 32.0;
};


