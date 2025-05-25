import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprimer les avertissements TensorFlow

from flask import Flask, render_template, request
from predict import Predictor  # ✅ Corrigé ici

app = Flask(__name__)

# Initialiser le prédicteur
predictor = Predictor()
predictor.load_models()  # Charger les modèles, les erreurs sont gérées dans Predictor

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    model_options = []
    selected_model = None

    if predictor.pytorch_model is not None:
        model_options.append(('pytorch', 'PyTorch'))
    if predictor.tensorflow_model is not None:
        model_options.append(('tensorflow', 'TensorFlow'))

    if not model_options:
        prediction = "Erreur : Aucun modèle disponible."

    if request.method == 'POST':
        selected_model = request.form['model']
        image_file = request.files['image']

        if image_file and image_file.filename != '':
            try:
                prediction = predictor.predict(image_file, framework=selected_model)
            except Exception as e:
                prediction = f"Erreur : {str(e)}"

    return render_template('index.html',
                           prediction=prediction,
                           model_options=model_options,
                           selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=True)
