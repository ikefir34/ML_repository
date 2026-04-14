import sys
import os
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

# Настройка путей
basedir = os.path.dirname(__file__)
ui_file_path = os.path.join(basedir, "untitled.ui") # Убедитесь, что имя совпадает

app = QApplication(sys.argv)

# Загрузка интерфейса
ui_file = QFile(ui_file_path)
if not ui_file.open(QFile.ReadOnly):
    print(f"Не удалось найти файл: {ui_file_path}")
    sys.exit(-1)

loader = QUiLoader()
# Загружаем интерфейс в объект диалога
window = loader.load(ui_file)
ui_file.close()

# Обработка кнопок (они уже встроены в QDialogButtonBox)
def on_accept():
    print("Нажато ОК")
    window.accept()

def on_reject():
    print("Нажато Отмена")
    window.reject()

# Подключаем сигналы стандартного блока кнопок
window.buttonBox.accepted.connect(on_accept)
window.buttonBox.rejected.connect(on_reject)

window.show()
sys.exit(app.exec())
