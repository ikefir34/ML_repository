import sys
from PySide6.QtCore import Qt, Slot, QEvent
from PySide6.QtGui import QKeyEvent, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, 
    QLabel, QLineEdit, QMessageBox
)

class EventMaster(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработка событий PySide6")
        self.resize(400, 300)

        # Виджеты
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Нажми любую клавишу или кликни мышкой")
        self.btn = QPushButton("Нажми меня (Сигнал)")
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.btn)

        # 1. СТАНДАРТНЫЕ СИГНАЛЫ (Высокоуровневые события)
        self.btn.clicked.connect(self.on_button_click)

    # --- 2. ПЕРЕОПРЕДЕЛЕНИЕ МЕТОДОВ (Низкоуровневые события) ---

    # Событие клика мыши по окну
    def mousePressEvent(self, event):
        pos = event.position()
        self.label.setText(f"Клик мышкой в: {pos.x():.0f}, {pos.y():.0f}")
        # Вызов базового класса (хорошая практика)
        super().mousePressEvent(event)

    # Событие нажатия клавиш на клавиатуре
    def keyPressEvent(self, event: QKeyEvent):
        # Проверяем, какая клавиша нажата
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            self.label.setText(f"Нажата клавиша: {event.text()}")

    # Событие закрытия окна (например, для подтверждения выхода)
    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(self, 'Выход', "Вы уверены?", 
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept() # Закрыть
        else:
            event.ignore() # Отменить закрытие

    # --- 3. СЛОТЫ ДЛЯ СИГНАЛОВ ---
    @Slot()
    def on_button_click(self):
        self.label.setText("Кнопка нажата через сигнал .clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EventMaster()
    window.show()
    sys.exit(app.exec())

    import sys
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, 
    QLabel, QLineEdit, QHBoxLayout, QMessageBox
)

class MyWindow(QWidget): # Наследуемся от QWidget (базовое окно)
    def __init__(self):
        super().__init__()

        # 1. ОСНОВНЫЕ НАСТРОЙКИ ОКНА
        self.setWindowTitle("Методичка PySide6")
        self.resize(400, 300)

        # 2. СОЗДАНИЕ ВИДЖЕТОВ (ЭЛЕМЕНТОВ ИНТЕРФЕЙСА)
        self.label = QLabel("Введите ваше имя:")
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Имя...")
        
        self.btn_greet = QPushButton("Поприветствовать")
        self.btn_clear = QPushButton("Очистить")

        # 3. МАКЕТЫ (LAYOUTS) — отвечают за расположение элементов
        # QVBoxLayout - вертикальный, QHBoxLayout - горизонтальный
        main_layout = QVBoxLayout() 
        button_layout = QHBoxLayout()

        # Добавляем виджеты в макеты
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.input_field)

        button_layout.addWidget(self.btn_greet)
        button_layout.addWidget(self.btn_clear)

        # Вкладываем один макет в другой
        main_layout.addLayout(button_layout)

        # Устанавливаем главный макет для окна
        self.setLayout(main_layout)

        # 4. СИГНАЛЫ И СЛОТЫ (СОБЫТИЯ)
        # Синтаксис: объект.сигнал.connect(функция_обработчик)
        self.btn_greet.clicked.connect(self.say_hello)
        self.btn_clear.clicked.connect(self.clear_input)

    # 5. ОБРАБОТЧИКИ СОБЫТИЙ (СЛОТЫ)
    @Slot() # Декоратор оптимизирует работу слота
    def say_hello(self):
        name = self.input_field.text().strip()
        if name:
            QMessageBox.information(self, "Приветствие", f"Привет, {name}!")
        else:
            self.label.setText("Эй, введи же что-нибудь!")
            self.label.setStyleSheet("color: red;")

    def clear_input(self):
        self.input_field.clear()
        self.label.setText("Введите ваше имя:")
        self.label.setStyleSheet("color: black;")

# 6. ЗАПУСК ПРИЛОЖЕНИЯ
if __name__ == "__main__":
    # Создаем объект приложения (обязательно один на программу)
    app = QApplication(sys.argv)

    # Создаем и показываем экземпляр нашего класса
    window = MyWindow()
    window.show()

    # Запускаем цикл обработки событий (event loop)
    sys.exit(app.exec())

