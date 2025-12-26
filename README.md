# Cat and Dog Image Classifier 🐱🐶

A deep learning project for binary image classification of cats and dogs using transfer learning with MobileNet.

---

## Language / Язык

- [English](#english-version) | [Русский](#русская-версия)

---

# English Version

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)

## Overview

This project implements a binary image classifier for cats and dogs using transfer learning. The model is based on MobileNet, a lightweight deep learning architecture pre-trained on ImageNet, fine-tuned for the specific task of classifying cats and dogs.

The project achieves **97.25% accuracy** on the test set, exceeding the 95% threshold required for the highest grade.

## Features

- ✅ Transfer learning with MobileNet pre-trained model
- ✅ Image augmentation for better generalization
- ✅ Proper train/validation/test split (70/15/15)
- ✅ Early stopping to prevent overfitting
- ✅ Comprehensive visualization of training metrics
- ✅ Multi-class classification setup (categorical)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook (optional, for running the notebook)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/himynameisartem/cat_dog_image_classifier.git
cd cat_dog_image_classifier
```

2. Install required packages:
```bash
pip install tensorflow numpy matplotlib jupyter
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the "Cats and Dogs" dataset, which contains:
- **Total images**: ~8,000 images
- **Classes**: 2 (cats and dogs)
- **Split**:
  - Training: 5,603 images (70%)
  - Validation: 1,201 images (15%)
  - Test: 1,201 images (15%)

The dataset is automatically downloaded from Yandex Cloud during execution.

## Model Architecture

The model uses the following architecture:

```
Input (160x160x3)
    ↓
MobileNet (pre-trained, frozen)
    ↓
GlobalAveragePooling2D()
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(2, activation='softmax')
```

**Key Features:**
- MobileNet base model (frozen weights)
- GlobalAveragePooling2D
- Dropout for regularization
- Softmax activation for multi-class classification

## Training

### Image Augmentation

The training data is augmented with:
- Rotation: ±40 degrees
- Width/Height shift: 20%
- Shear transformation: 20%
- Zoom: 20%
- Horizontal flip: enabled
- Fill mode: nearest

### Training Configuration

- **Optimizer**: Adam
- **Learning rate**: 1e-4
- **Loss function**: Categorical crossentropy
- **Batch size**: 20
- **Epochs**: 20 (with early stopping)
- **Early stopping**: Patience = 5 epochs, monitors validation accuracy

### Training Process

The model is trained with:
- Training data with augmentation
- Validation data without augmentation (for unbiased evaluation)
- Early stopping callback to restore best weights

## Results

### Performance Metrics

- **Training Accuracy**: ~96%
- **Validation Accuracy**: ~98.25%
- **Test Accuracy**: **97.25%** ✅

## Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook cat_dog_image_classifire.ipynb
```

2. Run all cells sequentially:
   - The dataset will be automatically downloaded
   - Data will be split into train/validation/test sets
   - Model will be created and compiled
   - Training will commence with progress visualization
   - Test accuracy will be evaluated

### Evaluating the Model

After training, the model can be evaluated on the test set:

```python
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc*100:.2f}%')
```

### Visualizing Results

The notebook includes visualization of:
- Training and validation loss curves
- Training and validation accuracy curves


# Русская Версия

## Оглавление

- [Обзор](#обзор)
- [Особенности](#особенности)
- [Требования](#требования)
- [Установка](#установка)
- [Датасет](#датасет)
- [Архитектура модели](#архитектура-модели)
- [Обучение](#обучение)
- [Результаты](#результаты)
- [Использование](#использование)

## Обзор

Этот проект реализует бинарный классификатор изображений кошек и собак с использованием трансферного обучения. Модель основана на MobileNet — легковесной архитектуре глубокого обучения, предобученной на ImageNet и дообученной для конкретной задачи классификации кошек и собак.

Проект достигает **97.25% точности** на тестовой выборке, превышая порог в 95%, необходимый для получения высшей оценки.

## Особенности

- ✅ Трансферное обучение с предобученной моделью MobileNet
- ✅ Аугментация изображений для лучшей генерализации
- ✅ Корректное разделение на train/validation/test (70/15/15)
- ✅ Early stopping для предотвращения переобучения
- ✅ Подробная визуализация метрик обучения
- ✅ Настройка многоклассовой классификации (categorical)

## Требования

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook (опционально, для запуска ноутбука)

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/himynameisartem/cat_dog_image_classifier.git
cd cat_dog_image_classifier
```

2. Установите необходимые пакеты:
```bash
pip install tensorflow numpy matplotlib jupyter
```

Или используйте файл requirements:
```bash
pip install -r requirements.txt
```

## Датасет

Проект использует датасет "Cats and Dogs", который содержит:
- **Всего изображений**: ~8,000 изображений
- **Классы**: 2 (кошки и собаки)
- **Разделение**:
  - Обучающая выборка: 5,603 изображения (70%)
  - Валидационная выборка: 1,201 изображение (15%)
  - Тестовая выборка: 1,201 изображение (15%)

Датасет автоматически загружается с Yandex Cloud во время выполнения.

## Архитектура модели

Модель использует следующую архитектуру:

```
Input (160x160x3)
    ↓
MobileNet (предобученная, замороженная)
    ↓
GlobalAveragePooling2D()
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(2, activation='softmax')
```

**Ключевые особенности:**
- Базовая модель MobileNet (замороженные веса)
- GlobalAveragePooling2D
- Dropout для регуляризации
- Softmax активация для многоклассовой классификации

## Обучение

### Аугментация изображений

Обучающие данные аугментируются с помощью:
- Поворот: ±40 градусов
- Сдвиг по ширине/высоте: 20%
- Сдвиг/искажение: 20%
- Масштабирование: 20%
- Горизонтальное отражение: включено
- Режим заполнения: nearest

### Конфигурация обучения

- **Оптимизатор**: Adam
- **Скорость обучения**: 1e-4
- **Функция потерь**: Categorical crossentropy
- **Размер батча**: 20
- **Эпохи**: 20 (с early stopping)
- **Early stopping**: Patience = 5 эпох, отслеживает валидационную точность

### Процесс обучения

Модель обучается с:
- Обучающими данными с аугментацией
- Валидационными данными без аугментации (для объективной оценки)
- Callback early stopping для восстановления лучших весов

## Результаты

### Метрики производительности

- **Точность на обучении**: ~96%
- **Точность на валидации**: ~98.25%
- **Точность на тесте**: **97.25%** ✅

## Использование

### Запуск ноутбука

1. Откройте Jupyter notebook:
```bash
jupyter notebook cat_dog_image_classifire.ipynb
```

2. Запустите все ячейки последовательно:
   - Датасет будет автоматически загружен
   - Данные будут разделены на train/validation/test выборки
   - Модель будет создана и скомпилирована
   - Начнется обучение с визуализацией прогресса
   - Будет оценена точность на тестовой выборке

### Оценка модели

После обучения модель может быть оценена на тестовой выборке:

```python
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc*100:.2f}%')
```

### Визуализация результатов

Ноутбук включает визуализацию:
- Кривых потерь на обучении и валидации
- Кривых точности на обучении и валидации
