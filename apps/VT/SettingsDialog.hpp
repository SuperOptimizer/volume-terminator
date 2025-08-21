#pragma once

#include <QDialog>
#include <QStringList>
#include <vector>

#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QPushButton>

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    SettingsDialog(QWidget* parent = nullptr);

    static std::vector<int> expandSettingToIntRange(const QString& setting);
    void updateVolumeList(const QStringList& volumeIds);

protected slots:
    void accept() override;

private:
    void setupUi();
    void loadSettings();
    void showTooltip(QPushButton* btn);

    // Volume Packages
    QLineEdit* edtDefaultPathVolpkg;
    QCheckBox* chkAutoOpenVolpkg;

    // Segment Viewer
    QSpinBox* spinFwdBackStepMs;
    QCheckBox* chkCenterOnZoom;
    QLineEdit* edtImpactRange;
    QLineEdit* edtScanRange;
    QSpinBox* spinScrollSpeed;
    QSpinBox* spinDisplayOpacity;
    QCheckBox* chkPlaySoundAfterSegRun;
    QLineEdit* edtUsername;
    QCheckBox* chkResetViewOnSurfaceChange;

    // Performance
    QSpinBox* spinPreloadedSlices;
    QCheckBox* chkSkipImageFormatConvExp;
    QSpinBox* spinParallelProcesses;
    QSpinBox* spinIterationCount;
    QComboBox* cmbDownscaleOverride;

    // Rendering
    QComboBox* cmbDefaultVolume;
    QLineEdit* edtOutputFormat;
    QDoubleSpinBox* spinScale;
    QSpinBox* spinResolution;
    QSpinBox* spinLayers;

    // Help buttons
    QPushButton* btnHelpScrollSpeed;
    QPushButton* btnHelpDisplayOpacity;
    QPushButton* btnHelpPreloadedSlices;
    QPushButton* btnHelpDownscaleOverride;
};