#include "formsetsrc.hpp"

FormSetSrc::FormSetSrc(Surface *op, QWidget* parent)
    : QWidget(parent)
{
    // Create widgets
    auto* layout = new QGridLayout(this);
    auto* label = new QLabel("Meshing Algorithm", this);
    _combo = new QComboBox(this);

    // Add combo items
    _combo->addItem("Raw");
    _combo->addItem("Blur");

    // Set up layout
    layout->addWidget(label, 0, 0);
    layout->addWidget(_combo, 0, 1);

    // Initialize chain
    _chain = dynamic_cast<OpChain*>(op);
    assert(_chain);

    _combo->setCurrentIndex(int(_chain->_src_mode));

    connect(_combo, &QComboBox::currentIndexChanged, this, &FormSetSrc::onAlgoIdxChanged);
}

void FormSetSrc::onAlgoIdxChanged(int index)
{
    if (!_chain)
        return;

    _chain->_src_mode = OpChainSourceMode(index);
    emit sendOpChainChanged(_chain);
}