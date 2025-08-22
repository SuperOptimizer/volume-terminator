#pragma once

#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QWidget>

class ConsoleOutputWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConsoleOutputWidget(QWidget* parent = nullptr);
    ~ConsoleOutputWidget() override;
    void appendOutput(const QString& text) const;

    void clear() const;
    void setTitle(const QString& title) const;


    void copyToClipboard() const;

private:
    QPlainTextEdit* _textEdit;
    QPushButton* _clearButton;
    QPushButton* _copyButton;
    QLabel* _titleLabel;
};

