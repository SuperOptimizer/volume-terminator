#pragma once

#include <QDockWidget>
#include "VCCollection.hpp"
#include <QTreeView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QWidget>
#include <QGroupBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QItemSelection>
#include <QDoubleSpinBox>




class CPointCollectionWidget : public QDockWidget
{
    Q_OBJECT

public:
    explicit CPointCollectionWidget(VCCollection *collection, QWidget *parent = nullptr);

signals:
    void collectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointDoubleClicked(uint64_t pointId);

public slots:
    void selectCollection(uint64_t collectionId) const;
    void selectPoint(uint64_t pointId) const;

private slots:
    void refreshTree() const;
    void onCollectionAdded(uint64_t collectionId) const;
    void onCollectionChanged(uint64_t collectionId) const;
    void onCollectionRemoved(uint64_t collectionId) const;
    void onPointAdded(const ColPoint& point) const;
    void onPointChanged(const ColPoint& point) const;
    void onPointRemoved(uint64_t pointId) const;

    void onResetClicked() const;
    void onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
    void onNewNameClicked() const;
    void onNameEdited(const QString &name) const;
    void onAbsoluteWindingChanged(int state) const;
    void onColorButtonClicked();
    void onWindingEdited(double value) const;
    void onWindingEnabledChanged(int state) const;
    void onFillWindingPlusClicked() const;
    void onFillWindingMinusClicked() const;
    void onFillWindingEqualsClicked() const;
    void onSaveClicked();
    void onLoadClicked();
  
 private:
    void keyPressEvent(QKeyEvent *event) override;
    void setupUi();
    void updateMetadataWidgets() const;
    QStandardItem* findCollectionItem(uint64_t collectionId) const;

    VCCollection *_point_collection = nullptr;
    uint64_t _selected_collection_id = 0;
    uint64_t _selected_point_id = 0;

    QTreeView *_tree_view;
    QStandardItemModel *_model;
 
    QPushButton *_load_button;
    QPushButton *_save_button;
    QPushButton *_reset_button;
 
    QGroupBox *_collection_metadata_group;
    QLineEdit *_collection_name_edit;
    QPushButton *_new_name_button;
    QCheckBox *_absolute_winding_checkbox;
    QPushButton *_color_button;
    QPushButton *_fill_winding_plus_button;
    QPushButton *_fill_winding_minus_button;
    QPushButton *_fill_winding_equals_button;

    QGroupBox *_point_metadata_group;
    QCheckBox *_winding_enabled_checkbox;
    QDoubleSpinBox* _winding_spinbox;
};


