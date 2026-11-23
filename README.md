# Brain Tumor Classification with CNNs

> **FR :** Projet de classification d'images cérébrales pour l'étude de tumeurs, avec des implémentations PyTorch et TensorFlow.  
> **EN:** Brain-image classification project exploring convolutional neural networks with PyTorch and TensorFlow.

## 🇬🇧 English

### Project overview
This repository contains an experimental deep-learning workflow for image classification. The command-line entry point, `main.py`, supports two frameworks (**PyTorch** and **TensorFlow**) and two modes: training and evaluation.

### Current capabilities
- Select the training framework with `--framework pytorch` or `--framework tensorflow`.
- Configure epochs, learning rate and weight decay.
- Request CUDA execution when available.
- Train a CNN and evaluate it on the test data.
- Save training history plots (provided PNG files).

### Repository layout
```text
.
├── main.py
├── data/
├── flask_app/
├── models/
├── training_history_pytorch.png
├── training_history_tensorflow.png
└── README.md
```

The data-loading and model implementation modules are located under `models/`. The exact dataset structure and class mapping should be checked in those modules before running the pipeline.

### Run
Create a virtual environment and install the frameworks and packages imported by the project (PyTorch, TensorFlow and the project's other dependencies). Then run:

```bash
python main.py --framework pytorch --mode train --epochs 10
python main.py --framework tensorflow --mode train --epochs 10
```

Evaluation is configured with `--mode eval`. The code expects trained weights at `models/thierno_model.torch` for PyTorch or `models/thierno_model.tensorflow.keras` for TensorFlow.

### Important
This is an educational/experimental image-classification project, not a medical diagnostic tool. Do not use its predictions for clinical decisions. Dataset provenance, patient privacy, class definitions and evaluation methodology should be documented before any research or deployment claim is made.

## 🇫🇷 Français

### Présentation
Ce dépôt présente un flux expérimental de classification d'images par réseaux de neurones convolutifs (CNN). Le fichier `main.py` permet de lancer l'entraînement ou l'évaluation avec **PyTorch** ou **TensorFlow**.

### Fonctionnalités repérées dans le code
- Choix du framework avec `--framework pytorch` ou `--framework tensorflow`.
- Paramétrage du nombre d'époques, du taux d'apprentissage et du weight decay.
- Utilisation éventuelle du GPU CUDA.
- Entraînement et évaluation du modèle.
- Visualisations d'historique d'entraînement fournies au format PNG.

### Structure
```text
.
├── main.py
├── data/
├── flask_app/
├── models/
├── training_history_pytorch.png
├── training_history_tensorflow.png
└── README.md
```

Les modules de préparation des données et de définition des modèles se trouvent dans `models/`. Vérifiez leur contenu pour connaître le format exact des données et les dépendances nécessaires.

### Exécution
Après installation de PyTorch, TensorFlow et des autres dépendances du projet :

```bash
python main.py --framework pytorch --mode train --epochs 10
python main.py --framework tensorflow --mode train --epochs 10
```

Le mode évaluation s'active avec `--mode eval`. Les poids doivent être présents aux emplacements attendus par le script.

### Avertissement
Ce projet est expérimental et pédagogique. Il ne constitue pas un dispositif médical et ses prédictions ne doivent jamais servir à établir un diagnostic ou à prendre une décision clinique.
