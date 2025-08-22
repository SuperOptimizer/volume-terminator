#pragma once

#include <QGraphicsItem>
#include <QPainterPath>
#include <QRectF>


class COutlinedTextItem : public QGraphicsTextItem
{
public:
    explicit COutlinedTextItem(QGraphicsItem *parent = nullptr);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    [[nodiscard]] QRectF boundingRect() const override;
    [[nodiscard]] QPainterPath shape() const override;
};

