#pragma once

#include <QWidget>
#include <QComboBox>
#include <QLabel>
#include <QGridLayout>
#include "OpChain.hpp"

class FormSetSrc : public QWidget
{
    Q_OBJECT

public:
    explicit FormSetSrc(Surface *op, QWidget* parent = nullptr);
    ~FormSetSrc() = default;

private slots:
    void onAlgoIdxChanged(int index);

    signals:
        void sendOpChainChanged(OpChain *chain);

private:
    OpChain *_chain;
    QComboBox *_combo;
};
