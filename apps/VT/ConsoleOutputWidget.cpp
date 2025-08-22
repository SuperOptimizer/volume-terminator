#include "ConsoleOutputWidget.hpp"
#include <QApplication>
#include <QClipboard>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QScrollBar>


ConsoleOutputWidget::ConsoleOutputWidget(QWidget* parent)
    : QWidget(parent)
    , _textEdit(new QPlainTextEdit(this))
    , _clearButton(new QPushButton(tr("Clear"), this))
    , _copyButton(new QPushButton(tr("Copy"), this))
    , _titleLabel(new QLabel(tr("Console Output"), this))
{

    auto* mainLayout = new QVBoxLayout(this);
    auto* headerLayout = new QHBoxLayout();
    auto* buttonLayout = new QHBoxLayout();

    _textEdit->setReadOnly(true);
    _textEdit->setLineWrapMode(QPlainTextEdit::NoWrap);

    QFont const consoleFont = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    _textEdit->setFont(consoleFont);
    
    _textEdit->setStyleSheet("QPlainTextEdit { background-color: #2b2b2b; color: #f0f0f0; }");

    _titleLabel->setStyleSheet("QLabel { font-weight: bold; }");
    headerLayout->addWidget(_titleLabel);
    headerLayout->addStretch(1);

    connect(_clearButton, &QPushButton::clicked, this, &ConsoleOutputWidget::clear);
    connect(_copyButton, &QPushButton::clicked, this, &ConsoleOutputWidget::copyToClipboard);
    
    buttonLayout->addWidget(_clearButton);
    buttonLayout->addWidget(_copyButton);
    buttonLayout->addStretch(1);

    mainLayout->addLayout(headerLayout);
    mainLayout->addWidget(_textEdit);
    mainLayout->addLayout(buttonLayout);
}

ConsoleOutputWidget::~ConsoleOutputWidget()
= default;

void ConsoleOutputWidget::appendOutput(const QString& text) const {
    _textEdit->appendPlainText(text);
    
    // auto-scroll 
    QScrollBar* scrollBar = _textEdit->verticalScrollBar();
    scrollBar->setValue(scrollBar->maximum());
}

void ConsoleOutputWidget::clear() const {
    _textEdit->clear();
}

void ConsoleOutputWidget::copyToClipboard() const {
    QApplication::clipboard()->setText(_textEdit->toPlainText());
}

void ConsoleOutputWidget::setTitle(const QString& title) const {
    _titleLabel->setText(title);
}

