#pragma once

#include <QWidget>
#include <QCheckBox>
#include <QGroupBox>
#include "OpChain.hpp"
#include "Surface.hpp"

class OpsSettings : public QWidget
{
    Q_OBJECT

public:
    explicit OpsSettings(QWidget* parent = nullptr);
    ~OpsSettings() = default;

public slots:
    void onOpSelected(Surface *op, OpChain *chain);
    void onEnabledChanged();

    signals:
        void sendOpChainChanged(OpChain *chain);

private:
    QGroupBox *_box;
    QCheckBox *_enable;
    Surface *_op = nullptr;
    OpChain *_chain = nullptr;
    QWidget *_form = nullptr;
};