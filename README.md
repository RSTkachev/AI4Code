# AI4Code

Проект для определения истинной последовательности ячеек Jupyter Notebook

## Структура проекта:
- Datasets - работа с данными:
    - cell.py - класс, для хранения токенизированной ячейки
    - cell_dataset.py - базовый класс датасета
    - train_val_cell_dataset.py - класс тренировочного и валидационного датасетов
    - sampler.py - сэмплер для тренировочного и валидационного даталоадеров
    - test_cell_dataset.py - класс тестового датасета
- utils.py - вычисление метрики Kendall Tau correlation и другие вспомогательные функции
- model.py - модель для определения порядка ячеек
- train.py - обучение модели, вычисление функции потерь и обновление параметров
- test.py - вычисление качества работы модели по метрике Kendall Tau correlation
- main.py - основной исполняемый файл