#pragma once

#include <QWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>

class ConsoleOutputWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConsoleOutputWidget(QWidget* parent = nullptr);
    ~ConsoleOutputWidget() override;
    void appendOutput(const QString& text) const;

    void clear() const;
    void setTitle(const QString& title) const;

public slots:
    void copyToClipboard() const;

private:
    QPlainTextEdit* _textEdit;
    QPushButton* _clearButton;
    QPushButton* _copyButton;
    QLabel* _titleLabel;
};

