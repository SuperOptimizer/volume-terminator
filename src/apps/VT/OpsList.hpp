#pragma once

#include <QWidget>
#include <QComboBox>
#include <z5/dataset.hxx>

#include "../../core/Slicing.hpp"
#include <QTreeWidgetItem>

#include "OpChain.hpp"

class OpsList : public QWidget
{
    Q_OBJECT

public:
    explicit OpsList(QWidget* parent = nullptr);
    ~OpsList() override;

    void setDataset(z5::Dataset *ds, ChunkCache *cache, float scale);


private slots:
    void onSelChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);

public slots:
    void onOpChainSelected(OpChain *ops);
    void onAppendOpClicked();

signals:
    void sendOpSelected(Surface *surf, OpChain *chain);
    void sendOpChainChanged(OpChain *chain);

private:
    QTreeWidget *_tree;
    QComboBox *_add_sel;
    OpChain *_op_chain = nullptr;

    //FIXME currently stored for refinement layer - make this somehow generic ...
    z5::Dataset *_ds = nullptr;
    ChunkCache *_cache = nullptr;
    float _scale = 0.0;
};
