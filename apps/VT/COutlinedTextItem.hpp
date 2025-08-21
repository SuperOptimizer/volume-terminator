#pragma once

#include <QGraphicsTextItem>
#include <QPainter>

namespace ChaoVis {

class COutlinedTextItem : public QGraphicsTextItem
{
public:
    explicit COutlinedTextItem(QGraphicsItem *parent = nullptr);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    QRectF boundingRect() const override;
    QPainterPath shape() const override;
};

}