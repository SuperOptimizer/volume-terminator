// SettingsDialog.cpp
#include "SettingsDialog.hpp"

#include <QSettings>
#include <QMessageBox>
#include <QToolTip>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QPushButton>
#include <QDialogButtonBox>

SettingsDialog::SettingsDialog(QWidget *parent) : QDialog(parent)
{
    setupUi();
    loadSettings();
}

void SettingsDialog::setupUi()
{
    setWindowTitle("Settings");
    resize(622, 580);

    QGridLayout* mainLayout = new QGridLayout(this);

    // Main group box container
    QGroupBox* mainGroup = new QGroupBox(this);
    QFormLayout* formLayout = new QFormLayout(mainGroup);

    // ========== Volume Packages Section ==========
    QGroupBox* volPkgGroup = new QGroupBox("Volume Packages", this);
    QGridLayout* volPkgLayout = new QGridLayout(volPkgGroup);

    QLabel* lblDefaultPath = new QLabel("Default *.volpkg path", this);
    edtDefaultPathVolpkg = new QLineEdit(this);

    chkAutoOpenVolpkg = new QCheckBox("Auto open last used *.volpkg upon app start", this);
    chkAutoOpenVolpkg->setChecked(true);

    volPkgLayout->addWidget(lblDefaultPath, 0, 0);
    volPkgLayout->addWidget(edtDefaultPathVolpkg, 0, 1);
    volPkgLayout->addWidget(chkAutoOpenVolpkg, 1, 0, 1, 2);

    formLayout->addRow(volPkgGroup);

    // ========== Segment Viewer Section ==========
    QGroupBox* segViewerGroup = new QGroupBox("Segment Viewer", this);
    QGridLayout* segViewerLayout = new QGridLayout(segViewerGroup);

    // Row 0: Forward/Backward step
    QLabel* lblFwdBack = new QLabel("Mouse Forward/Backwards: Milliseconds delay", this);
    spinFwdBackStepMs = new QSpinBox(this);
    spinFwdBackStepMs->setMinimum(25);
    spinFwdBackStepMs->setMaximum(1000);
    spinFwdBackStepMs->setValue(25);

    // Row 1: Center on zoom
    chkCenterOnZoom = new QCheckBox("Center on cursor when zooming via mouse wheel", this);

    // Row 2: Impact range
    QLabel* lblImpactRange = new QLabel("Impact range steps for A/D and Mouse Wheel + W", this);
    edtImpactRange = new QLineEdit("1-3, 5, 8, 11, 15, 20, 28, 40, 60, 100, 200", this);

    // Row 3: Scan range
    QLabel* lblScanRange = new QLabel("Slice scanning steps for Mouse Wheel + Shift", this);
    edtScanRange = new QLineEdit("1, 2, 5, 10, 20, 50, 100, 200, 500, 1000", this);

    // Row 4: Scroll speed
    QLabel* lblScrollSpeed = new QLabel("Scroll speed (-1 = use default)", this);
    spinScrollSpeed = new QSpinBox(this);
    spinScrollSpeed->setMinimum(-1);
    spinScrollSpeed->setValue(-1);

    btnHelpScrollSpeed = new QPushButton(this);
    btnHelpScrollSpeed->setFlat(true);
    btnHelpScrollSpeed->setIcon(QIcon::fromTheme("help-contents"));
    btnHelpScrollSpeed->setToolTip("<html><head/><body><p>-1 and 0 = default speed </p>"
                                   "<p>1 = very slow/fine scrolling</p>"
                                   "<p>higher values = faster scrolling</p></body></html>");

    // Row 5: Display opacity
    QLabel* lblDisplayOpacity = new QLabel("Opacity of \"Display\" mode segments", this);
    spinDisplayOpacity = new QSpinBox(this);
    spinDisplayOpacity->setSuffix("%");
    spinDisplayOpacity->setMinimum(0);
    spinDisplayOpacity->setMaximum(100);
    spinDisplayOpacity->setValue(70);

    btnHelpDisplayOpacity = new QPushButton(this);
    btnHelpDisplayOpacity->setFlat(true);
    btnHelpDisplayOpacity->setIcon(QIcon::fromTheme("help-contents"));
    btnHelpDisplayOpacity->setToolTip("<html><head/><body><p>Applies to non highlighted segments in display mode</p>"
                                      "<p>0% = completely transparent = invisible</p>"
                                      "<p>100% = completely opaque</p></body></html>");

    // Row 6: Play sound
    chkPlaySoundAfterSegRun = new QCheckBox("Play sound when segmentation run finished", this);
    chkPlaySoundAfterSegRun->setChecked(true);

    // Row 7: Username
    QLabel* lblUsername = new QLabel("Username (for tagging)", this);
    edtUsername = new QLineEdit(this);
    edtUsername->setPlaceholderText("Enter your username");

    // Row 8: Reset view
    chkResetViewOnSurfaceChange = new QCheckBox("Reset view when switching between surfaces", this);
    chkResetViewOnSurfaceChange->setChecked(true);
    chkResetViewOnSurfaceChange->setToolTip("When enabled, the view will automatically fit and center "
                                            "on each surface when switching between segments");

    // Add widgets to segment viewer layout
    segViewerLayout->addWidget(lblFwdBack, 0, 0);
    segViewerLayout->addWidget(spinFwdBackStepMs, 0, 1);
    segViewerLayout->addWidget(chkCenterOnZoom, 1, 0);
    segViewerLayout->addWidget(lblImpactRange, 2, 0);
    segViewerLayout->addWidget(edtImpactRange, 2, 1);
    segViewerLayout->addWidget(lblScanRange, 3, 0);
    segViewerLayout->addWidget(edtScanRange, 3, 1);
    segViewerLayout->addWidget(lblScrollSpeed, 4, 0);
    segViewerLayout->addWidget(spinScrollSpeed, 4, 1);
    segViewerLayout->addWidget(btnHelpScrollSpeed, 4, 2);
    segViewerLayout->addWidget(lblDisplayOpacity, 5, 0);
    segViewerLayout->addWidget(spinDisplayOpacity, 5, 1);
    segViewerLayout->addWidget(btnHelpDisplayOpacity, 5, 2);
    segViewerLayout->addWidget(chkPlaySoundAfterSegRun, 6, 0);
    segViewerLayout->addWidget(lblUsername, 7, 0);
    segViewerLayout->addWidget(edtUsername, 7, 1);
    segViewerLayout->addWidget(chkResetViewOnSurfaceChange, 8, 0, 1, 2);

    formLayout->addRow(segViewerGroup);

    // ========== Performance Section ==========
    QGroupBox* perfGroup = new QGroupBox("Performance", this);
    QGridLayout* perfLayout = new QGridLayout(perfGroup);
    perfLayout->setColumnStretch(0, 1);
    perfLayout->setColumnStretch(1, 1);

    // Preloaded slices
    QLabel* lblPreloadedSlices = new QLabel("Number of preloaded slices", this);
    spinPreloadedSlices = new QSpinBox(this);
    spinPreloadedSlices->setMinimum(1);
    spinPreloadedSlices->setMaximum(9999);
    spinPreloadedSlices->setValue(200);

    btnHelpPreloadedSlices = new QPushButton(this);
    btnHelpPreloadedSlices->setFlat(true);
    btnHelpPreloadedSlices->setIcon(QIcon::fromTheme("help-contents"));
    btnHelpPreloadedSlices->setToolTip("<html><head/><body><p>Number of slices that will be preloaded into RAM "
                                       "once you enter the Segmentation Tool. </p>"
                                       "<p>They are centered around the slice the Segmentation Tool was started on "
                                       "(half above, half below).</p></body></html>");

    // Skip image format conversion
    chkSkipImageFormatConvExp = new QCheckBox("Skip image format conversion (experimental)", this);

    // Parallel processes
    QLabel* lblParallelProcesses = new QLabel("Num processes for xargs in CommandLineRunner", this);
    spinParallelProcesses = new QSpinBox(this);
    spinParallelProcesses->setMinimum(1);
    spinParallelProcesses->setMaximum(128);
    spinParallelProcesses->setValue(8);

    // Iteration count
    QLabel* lblIterationCount = new QLabel("Num seq for xargs in CommandLineRunner", this);
    spinIterationCount = new QSpinBox(this);
    spinIterationCount->setMinimum(1);
    spinIterationCount->setMaximum(50000000);
    spinIterationCount->setValue(1000);

    // Downscale override
    QLabel* lblDownscaleOverride = new QLabel("Force downscale level (reduces quality for performance)", this);
    cmbDownscaleOverride = new QComboBox(this);
    cmbDownscaleOverride->addItems({"1x (full res)", "2x", "4x", "8x", "16x", "32x"});

    btnHelpDownscaleOverride = new QPushButton(this);
    btnHelpDownscaleOverride->setFlat(true);
    btnHelpDownscaleOverride->setIcon(QIcon::fromTheme("help-contents"));
    btnHelpDownscaleOverride->setToolTip("<html><head/><body><p>Forces the viewer to use a more downscaled version of the zarr volume.</p>"
                                         "<p>This reduces image quality but improves performance and reduces file I/O.</p>"
                                         "<p>1x (full res) = Use appropriate scale for zoom level<br/>"
                                         "2x = Use 2x more downscaled data<br/>"
                                         "4x = Use 4x more downscaled data<br/>etc.</p></body></html>");

    // Add widgets to performance layout
    perfLayout->addWidget(lblPreloadedSlices, 0, 0);
    perfLayout->addWidget(spinPreloadedSlices, 0, 1);
    perfLayout->addWidget(btnHelpPreloadedSlices, 0, 2);
    perfLayout->addWidget(chkSkipImageFormatConvExp, 1, 0);
    perfLayout->addWidget(lblParallelProcesses, 2, 0);
    perfLayout->addWidget(spinParallelProcesses, 2, 1);
    perfLayout->addWidget(lblIterationCount, 3, 0);
    perfLayout->addWidget(spinIterationCount, 3, 1);
    perfLayout->addWidget(lblDownscaleOverride, 4, 0);
    perfLayout->addWidget(cmbDownscaleOverride, 4, 1);
    perfLayout->addWidget(btnHelpDownscaleOverride, 4, 2);

    formLayout->addRow(perfGroup);

    // ========== Rendering Section ==========
    QGroupBox* renderGroup = new QGroupBox("Rendering", this);
    QGridLayout* renderLayout = new QGridLayout(renderGroup);

    // Default volume
    QLabel* lblDefaultVolume = new QLabel("Default Target Volume", this);
    cmbDefaultVolume = new QComboBox(this);

    // Output format
    QLabel* lblOutputFormat = new QLabel("Output Path Format", this);
    edtOutputFormat = new QLineEdit("%s/layers/%02d.tif", this);

    // Scale
    QLabel* lblScale = new QLabel("Default Scale", this);
    spinScale = new QDoubleSpinBox(this);
    spinScale->setMinimum(0.01);
    spinScale->setMaximum(10.0);
    spinScale->setSingleStep(0.1);
    spinScale->setValue(1.0);

    // Resolution
    QLabel* lblResolution = new QLabel("Default Resolution", this);
    spinResolution = new QSpinBox(this);
    spinResolution->setMinimum(0);
    spinResolution->setMaximum(5);
    spinResolution->setValue(0);

    // Layers
    QLabel* lblLayers = new QLabel("Default Layers", this);
    spinLayers = new QSpinBox(this);
    spinLayers->setMinimum(1);
    spinLayers->setMaximum(100);
    spinLayers->setValue(21);

    // Add widgets to rendering layout
    renderLayout->addWidget(lblDefaultVolume, 0, 0);
    renderLayout->addWidget(cmbDefaultVolume, 0, 1);
    renderLayout->addWidget(lblOutputFormat, 1, 0);
    renderLayout->addWidget(edtOutputFormat, 1, 1);
    renderLayout->addWidget(lblScale, 2, 0);
    renderLayout->addWidget(spinScale, 2, 1);
    renderLayout->addWidget(lblResolution, 3, 0);
    renderLayout->addWidget(spinResolution, 3, 1);
    renderLayout->addWidget(lblLayers, 4, 0);
    renderLayout->addWidget(spinLayers, 4, 1);

    formLayout->addRow(renderGroup);

    // ========== Dialog Buttons ==========
    QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                                                       Qt::Horizontal, this);

    mainLayout->addWidget(mainGroup, 0, 0);
    mainLayout->addWidget(buttonBox, 1, 0);

    // Connect signals
    connect(buttonBox, &QDialogButtonBox::accepted, this, &SettingsDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // Connect help buttons
    connect(btnHelpDownscaleOverride, &QPushButton::clicked,
            [this]{ showTooltip(btnHelpDownscaleOverride); });
    connect(btnHelpScrollSpeed, &QPushButton::clicked,
            [this]{ showTooltip(btnHelpScrollSpeed); });
    connect(btnHelpDisplayOpacity, &QPushButton::clicked,
            [this]{ showTooltip(btnHelpDisplayOpacity); });
    connect(btnHelpPreloadedSlices, &QPushButton::clicked,
            [this]{ showTooltip(btnHelpPreloadedSlices); });
}

void SettingsDialog::loadSettings() const {
    QSettings settings("VC.ini", QSettings::IniFormat);

    edtDefaultPathVolpkg->setText(settings.value("volpkg/default_path").toString());
    chkAutoOpenVolpkg->setChecked(settings.value("volpkg/auto_open", true).toInt() != 0);

    spinFwdBackStepMs->setValue(settings.value("viewer/fwd_back_step_ms", 25).toInt());
    chkCenterOnZoom->setChecked(settings.value("viewer/center_on_zoom", false).toInt() != 0);
    edtImpactRange->setText(settings.value("viewer/impact_range_steps", "1-3, 5, 8, 11, 15, 20, 28, 40, 60, 100, 200").toString());
    edtScanRange->setText(settings.value("viewer/scan_range_steps", "1, 2, 5, 10, 20, 50, 100, 200, 500, 1000").toString());
    spinScrollSpeed->setValue(settings.value("viewer/scroll_speed", -1).toInt());
    spinDisplayOpacity->setValue(settings.value("viewer/display_segment_opacity", 70).toInt());
    chkPlaySoundAfterSegRun->setChecked(settings.value("viewer/play_sound_after_seg_run", true).toInt() != 0);
    edtUsername->setText(settings.value("viewer/username", "").toString());
    chkResetViewOnSurfaceChange->setChecked(settings.value("viewer/reset_view_on_surface_change", true).toInt() != 0);

    spinPreloadedSlices->setValue(settings.value("perf/preloaded_slices", 200).toInt());
    chkSkipImageFormatConvExp->setChecked(settings.value("perf/chkSkipImageFormatConvExp", false).toBool());
    spinParallelProcesses->setValue(settings.value("perf/parallel_processes", 8).toInt());
    spinIterationCount->setValue(settings.value("perf/iteration_count", 1000).toInt());
    cmbDownscaleOverride->setCurrentIndex(settings.value("perf/downscale_override", 0).toInt());

    // Load rendering settings
    QString defaultVolume = settings.value("rendering/default_volume", "").toString();
    cmbDefaultVolume->addItem(""); // Empty selection
    cmbDefaultVolume->setCurrentText(defaultVolume);

    edtOutputFormat->setText(settings.value("rendering/output_path_format", "%s/layers/%02d.tif").toString());
    spinScale->setValue(settings.value("rendering/scale", 1.0).toDouble());
    spinResolution->setValue(settings.value("rendering/resolution", 0).toInt());
    spinLayers->setValue(settings.value("rendering/layers", 21).toInt());
}

void SettingsDialog::accept()
{
    // Store the settings
    QSettings settings("VC.ini", QSettings::IniFormat);

    settings.setValue("volpkg/default_path", edtDefaultPathVolpkg->text());
    settings.setValue("volpkg/auto_open", chkAutoOpenVolpkg->isChecked() ? "1" : "0");

    settings.setValue("viewer/fwd_back_step_ms", spinFwdBackStepMs->value());
    settings.setValue("viewer/center_on_zoom", chkCenterOnZoom->isChecked() ? "1" : "0");
    settings.setValue("viewer/impact_range_steps", edtImpactRange->text());
    settings.setValue("viewer/scan_range_steps", edtScanRange->text());
    settings.setValue("viewer/scroll_speed", spinScrollSpeed->value());
    settings.setValue("viewer/display_segment_opacity", spinDisplayOpacity->value());
    settings.setValue("viewer/play_sound_after_seg_run", chkPlaySoundAfterSegRun->isChecked() ? "1" : "0");
    settings.setValue("viewer/username", edtUsername->text());
    settings.setValue("viewer/reset_view_on_surface_change", chkResetViewOnSurfaceChange->isChecked() ? "1" : "0");

    settings.setValue("perf/preloaded_slices", spinPreloadedSlices->value());
    settings.setValue("perf/chkSkipImageFormatConvExp", chkSkipImageFormatConvExp->isChecked() ? "1" : "0");
    settings.setValue("perf/parallel_processes", spinParallelProcesses->value());
    settings.setValue("perf/iteration_count", spinIterationCount->value());
    settings.setValue("perf/downscale_override", cmbDownscaleOverride->currentIndex());

    // Store rendering settings
    settings.setValue("rendering/default_volume", cmbDefaultVolume->currentText());
    settings.setValue("rendering/output_path_format", edtOutputFormat->text());
    settings.setValue("rendering/scale", spinScale->value());
    settings.setValue("rendering/resolution", spinResolution->value());
    settings.setValue("rendering/layers", spinLayers->value());

    QMessageBox::information(this, tr("Restart required"),
                           tr("Note: Some settings only take effect once you restarted the app."));

    close();
}

// Expand string that contains a range definition from the user settings into an integer vector
std::vector<int> SettingsDialog::expandSettingToIntRange(const QString& setting)
{
    std::vector<int> res;
    if (setting.isEmpty()) {
        return res;
    }

    auto value = setting.simplified();
    value.replace(" ", "");
    auto commaSplit = value.split(",");
    for(auto str : commaSplit) {
        if (str.contains("-")) {
            // Expand the range to distinct values
            auto dashSplit = str.split("-");
            // We need to have two split results (before and after the dash), otherwise skip
            if (dashSplit.size() == 2) {
                for(int i = dashSplit.at(0).toInt(); i <= dashSplit.at(1).toInt(); i++) {
                    res.push_back(i);
                }
            }
        } else {
            res.push_back(str.toInt());
        }
    }

    return res;
}

void SettingsDialog::updateVolumeList(const QStringList& volumeIds) const {
    QString currentVolume = cmbDefaultVolume->currentText();
    cmbDefaultVolume->clear();

    // Always add an empty option
    cmbDefaultVolume->addItem("");

    for (const QString& id : volumeIds) {
        cmbDefaultVolume->addItem(id);
    }

    // Try to restore the previous selection
    int index = cmbDefaultVolume->findText(currentVolume);
    if (index >= 0) {
        cmbDefaultVolume->setCurrentIndex(index);
    }
}

void SettingsDialog::showTooltip(const QPushButton* btn)
{
    QToolTip::showText(QCursor::pos(), btn->toolTip());
}