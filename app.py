from flask import Flask, request, render_template, jsonify
import os
import pickle
import traceback
import sys

# Try to import keras/tf needed for the model
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    has_tf = True
except ImportError:
    has_tf = False
    print("TensorFlow not found. Model loading will be disabled.", file=sys.stderr)

app = Flask(__name__)

MODEL_PATH = "bilstm_model (1).h5"
TOKENIZER_PATH = "tokenizer (1).pkl"

model = None
tokenizer = None

def load_resources():
    global model, tokenizer
    if has_tf and model is None:
        try:
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
                print("Model loaded successfully.", file=sys.stderr)
            else:
                print(f"Model file not found: {MODEL_PATH}", file=sys.stderr)
                
            if os.path.exists(TOKENIZER_PATH):
                with open(TOKENIZER_PATH, 'rb') as f:
                    tokenizer = pickle.load(f)
                print("Tokenizer loaded successfully.", file=sys.stderr)
            else:
                print(f"Tokenizer file not found: {TOKENIZER_PATH}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading resources: {e}", file=sys.stderr)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    sequence = ""
    
    if request.method == "POST":
        sequence = request.form.get("sequence", "").strip()
        if not sequence:
            error = "Please enter a valid protein sequence."
        else:
            load_resources()
            if model is not None and tokenizer is not None:
                try:
                    # Basic preprocessing
                    seqs = tokenizer.texts_to_sequences([sequence])
                    
                    try:
                        input_shape = model.input_shape
                        maxlen = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else 1000
                    except Exception:
                        maxlen = 1000
                        
                    padded = pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post')
                    
                    preds = model.predict(padded)
                    
                    # Assuming binary classification
                    if preds.shape[-1] == 1:
                        prob = float(preds[0][0])
                        if prob >= 0.5:
                            prediction = f"Likely to crystallize ({prob:.2%} probability)"
                        else:
                            prediction = f"Unlikely to crystallize ({prob:.2%} probability)"
                    else:
                        # Fallback for multi-class
                        class_idx = int(np.argmax(preds[0]))
                        prob = float(preds[0][class_idx])
                        prediction = f"Predicted Class: {class_idx} ({prob:.2%} probability)"
                        
                except Exception as e:
                    traceback.print_exc()
                    error = f"Error during prediction: {str(e)}"
            else:
                error = "Unable to process prediction. Please ensure TensorFlow is installed and model files exist."
                
    return render_template("index.html", prediction=prediction, error=error, sequence=sequence)

if __name__ == "__main__":
    # Load resources at startup if possible
    load_resources()
    app.run(debug=True, port=5000)
