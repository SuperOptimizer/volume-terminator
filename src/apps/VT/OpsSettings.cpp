#include "OpsSettings.hpp"

#include "OpChain.hpp"
#include "formsetsrc.hpp"

#include <iostream>

#include "OpsList.hpp"

#include "OpsSettings.hpp"
#include "OpChain.hpp"
#include "formsetsrc.hpp"
#include <QVBoxLayout>
#include <QGridLayout>

OpsSettings::OpsSettings(QWidget* parent)
    : QWidget(parent)
{
    // Create main layout
    auto* mainLayout = new QGridLayout(this);

    // Create group box
    _box = new QGroupBox(this);
    _box->setVisible(false);

    // Create checkbox inside group box
    auto* boxLayout = new QVBoxLayout(_box);
    _enable = new QCheckBox("Layer enabled", _box);
    _enable->setChecked(true);
    boxLayout->addWidget(_enable);

    // Add group box to main layout
    mainLayout->addWidget(_box, 0, 0);

    connect(_enable, &QCheckBox::stateChanged, this, &OpsSettings::onEnabledChanged);
}


void OpsSettings::onEnabledChanged()
{
    if (_chain) {
        _chain->setEnabled((DeltaSurface*)_op, _enable->isChecked());
        sendOpChainChanged(_chain);
    }
}

QWidget *op_form_widget(Surface *op, OpsSettings *parent)
{
    if (!op)
        return nullptr;
    
    if (dynamic_cast<OpChain*>(op)) {
        auto w = new FormSetSrc(op, parent);
        QWidget::connect(w, &FormSetSrc::sendOpChainChanged,
                        parent, &OpsSettings::sendOpChainChanged);
        return w;
    }

    return nullptr;
}


void OpsSettings::onOpSelected(Surface *op, OpChain *chain)
{
    _op = op;
    _chain = chain;

    // If we have no layer selected (e.g. because a new surface was selected to
    // display which resets the layer selection), hide the box until a
    // layer actually is selected.
    if(!_op) {
        _box->setVisible(false);
        return;
    }

    _box->setTitle(tr("Selected Layer: %1").arg(QString(op_name(op))));    

    if (!dynamic_cast<DeltaSurface*>(_op))
        _enable->setEnabled(false);
    else {
        _enable->setEnabled(true);
        QSignalBlocker blocker(_enable);
        _enable->setChecked(_chain->enabled((DeltaSurface*)_op));
    }
    
    if (_form)
        delete _form;
    
    _form = op_form_widget(op, this);
    if (_form) {
        _box->layout()->addWidget(_form);
        _box->setVisible(true);
    }
}
