#include "OpsList.hpp"

#include "OpChain.hpp"

#include <iostream>

#include <QComboBox>

#include "OpsList.hpp"
#include "OpChain.hpp"
#include <QTreeWidget>
#include <QComboBox>
#include <QPushButton>
#include <QGridLayout>
#include <QHBoxLayout>

OpsList::OpsList(QWidget* parent) : QWidget(parent)
{
    // Create widgets
    auto* layout = new QGridLayout(this);

    _tree = new QTreeWidget(this);
    _tree->setAlternatingRowColors(true);
    _tree->setRootIsDecorated(false);
    _tree->setHeaderHidden(true);

    auto* buttonLayout = new QHBoxLayout();
    auto* appendButton = new QPushButton("Append", this);
    _add_sel = new QComboBox(this);
    _add_sel->addItem("refineAlphaComp");

    buttonLayout->addWidget(appendButton);
    buttonLayout->addWidget(_add_sel);

    // Set up layout
    layout->addWidget(_tree, 0, 0);
    layout->addLayout(buttonLayout, 1, 0);

    // Connect signals
    connect(_tree, &QTreeWidget::currentItemChanged, this, &OpsList::onSelChanged);
    connect(appendButton, &QPushButton::clicked, this, &OpsList::onAppendOpClicked);
}

OpsList::~OpsList() {  }

void OpsList::onOpChainSelected(OpChain *ops)
{
    _op_chain = ops;
    std::cout << "OpChain/Layer selected " << ops << "\n";

    _tree->clear();

    if (_op_chain) {
        QTreeWidgetItem *item = new QTreeWidgetItem(_tree);
        item->setText(0, QString(op_name(_op_chain)));
        item->setData(0, Qt::UserRole, QVariant::fromValue((void*)_op_chain));

        for (auto& op : _op_chain->ops()) {
            QTreeWidgetItem *item = new QTreeWidgetItem(_tree);
            item->setText(0, QString(op_name(op)));
            item->setData(0, Qt::UserRole, QVariant::fromValue((void*)op));
        }
    }
}

void OpsList::onSelChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    if (!current)
        sendOpSelected(nullptr, _op_chain);
    else
        sendOpSelected((Surface*)qvariant_cast<void*>(current->data(0, Qt::UserRole)), _op_chain);
}

void OpsList::onAppendOpClicked()
{
    if (!_op_chain)
        return;

    QString sel = _add_sel->currentText();

    if (sel == "refineAlphaComp") {
        assert(_cache);
        assert(_ds);
        _op_chain->append(new RefineCompSurface(_ds, _cache));
    }

    onOpChainSelected(_op_chain);
    sendOpChainChanged(_op_chain);
}

void OpsList::setDataset(z5::Dataset *ds, ChunkCache *cache, float scale)
{
    _ds = ds;
    _cache = cache;
    _scale = scale;
}
