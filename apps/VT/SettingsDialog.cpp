// SettingsDialog.cpp
// Philip Allgaier 2023 October
#include "SettingsDialog.hpp"

#include <QSettings>
#include <QMessageBox>
#include <QToolTip>


SettingsDialog::SettingsDialog(QWidget *parent) : QDialog(parent)
{
    setupUi(this);

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
    // Note: The combobox will be populated with actual volumes when a volpkg is loaded
    cmbDefaultVolume->setCurrentText(defaultVolume);
    
    edtOutputFormat->setText(settings.value("rendering/output_path_format", "%s/layers/%02d.tif").toString());
    spinScale->setValue(settings.value("rendering/scale", 1.0).toDouble());
    spinResolution->setValue(settings.value("rendering/resolution", 0).toInt());
    spinLayers->setValue(settings.value("rendering/layers", 21).toInt());


    connect(btnHelpDownscaleOverride, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpDownscaleOverride->toolTip()); });
    connect(btnHelpScrollSpeed, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpScrollSpeed->toolTip()); });
    connect(btnHelpDisplayOpacity, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpDisplayOpacity->toolTip()); });
    connect(btnHelpPreloadedSlices, &QPushButton::clicked, this, [this]{ QToolTip::showText(QCursor::pos(), btnHelpPreloadedSlices->toolTip()); });
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

    QMessageBox::information(this, tr("Restart required"), tr("Note: Some settings only take effect once you restarted the app."));

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

void SettingsDialog::updateVolumeList(const QStringList& volumeIds)
{
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
