import torch
import tensorflow as tf
import os
from PIL import Image
from torchvision import transforms
import sys
import numpy as np  # Importé pour utiliser np.transpose

# Ajouter le répertoire racine du projet au chemin Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Importer les fonctions de définition des modèles
from models.cnn_pytorch import get_pytorch_model
from models.cnn_tensorflow import get_tensorflow_model

class Predictor:
    def __init__(self):
        """Initialiser le prédicteur avec les paramètres de base."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pytorch_model = None  # Modèle PyTorch, chargé si disponible
        self.tensorflow_model = None  # Modèle TensorFlow, chargé si disponible
        self.class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']  # Classes corrigées

    def load_models(self):
        """Charger les modèles PyTorch et TensorFlow depuis le répertoire models."""
        model_dir = os.path.join(project_root, 'models')
        try:
            self._load_pytorch_model(model_dir)
        except FileNotFoundError as e:
            print(f"Erreur PyTorch : {e}")
            self.pytorch_model = None
        try:
            self._load_tensorflow_model(model_dir)
        except FileNotFoundError as e:
            print(f"Erreur TensorFlow : {e}")
            self.tensorflow_model = None

    def _load_pytorch_model(self, model_dir):
        """Charger le modèle PyTorch à partir du chemin spécifié."""
        model_path = os.path.join(model_dir, 'thierno_model.torch')
        if os.path.exists(model_path):
            model = get_pytorch_model()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.pytorch_model = model.to(self.device)
        else:
            raise FileNotFoundError(f"Modèle PyTorch non trouvé à : {model_path}")

    def _load_tensorflow_model(self, model_dir):
        """Charger le modèle TensorFlow à partir du chemin spécifié."""
        model_path = os.path.join(model_dir, 'thierno_model.tensorflow.keras')
        if os.path.exists(model_path):
            self.tensorflow_model = tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Modèle TensorFlow non trouvé à : {model_path}")

    def preprocess_image(self, image):
        """Pré-traiter l'image pour qu'elle soit compatible avec les modèles."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Redimensionner à 224x224
            transforms.ToTensor(),          # Convertir en tenseur
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # Normalisation pour 3 canaux
        ])
        image = transform(image).unsqueeze(0)  # Ajouter une dimension par lot
        return image

    def predict(self, image_file, framework="pytorch"):
        """Effectuer une prédiction sur une image en utilisant le framework spécifié."""
        if not image_file or not hasattr(image_file, 'read'):
            raise FileNotFoundError("Aucune image fournie ou fichier invalide")
        
        # Ouvrir et convertir l'image en RGB
        image = Image.open(image_file).convert('RGB')
        image_processed = self.preprocess_image(image)
        
        if framework == "pytorch" and self.pytorch_model is not None:
            image_processed = image_processed.to(self.device)
            with torch.no_grad():
                output = self.pytorch_model(image_processed)
                pred_idx = torch.argmax(output, dim=1).item()
        elif framework == "tensorflow" and self.tensorflow_model is not None:
            image_array = image_processed.numpy()
            image_array = np.transpose(image_array, (0, 2, 3, 1))  # Ajuster les dimensions pour TensorFlow
            output = self.tensorflow_model.predict(image_array)
            pred_idx = np.argmax(output, axis=1)[0]
        else:
            raise ValueError("Framework non supporté ou modèle non chargé")
        
        return self.class_labels[pred_idx]
