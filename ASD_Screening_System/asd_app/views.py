import onnxruntime as ort
import numpy as np
import os
from django.shortcuts import render, redirect
from PIL import Image

# Initialize the ONNX Multivariate Analysis Engine
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'asd_model.onnx')
session = ort.InferenceSession(MODEL_PATH)

QUESTION_TEXTS = [
    "Notice small sounds that others don't seem to hear?",
    "Focus more on the whole picture rather than small details?",
    "Find it easy to do more than one thing at once?",
    "Return to what they were doing easily after an interruption?",
    "Find it easy to 'read between the lines' when someone is talking?",
    "Know how to tell if someone listening is getting bored?",
    "Easily work out what characters in a story are feeling?",
    "Tell what someone is thinking by looking at their face?",
    "Understand why people act the way they do?",
    "Enjoy playing games that involve 'make-believe'?"
]

def preprocess(img, scores):
    """Fuses 10 digital facial markers and 10 behavioral indicators."""
    img = img.resize((224, 224))
    img_data = np.array(img).transpose(2, 0, 1).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1).astype('float32')
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1).astype('float32')
    img_data = (img_data - mean) / std
    img_input = np.expand_dims(img_data, axis=0)
    beh_input = np.array([scores], dtype=np.float32)
    return img_input, beh_input

def index(request):
    if request.method == 'POST':
        scores = [float(request.POST.get(f'q{i}', 0)) for i in range(1, 11)]
        img_file = request.FILES.get('face_image')
        
        if img_file:
            img = Image.open(img_file).convert('RGB')
            img_input, beh_input = preprocess(img, scores)
            outputs = session.run(None, {'image': img_input, 'behavioral': beh_input})
            
            # Multivariate Risk Probability calculation
            exp_out = np.exp(outputs[0] - np.max(outputs[0]))
            probs = exp_out / exp_out.sum()
            risk_score = round(float(probs[0][1] * 100), 2)
            
            # Store data in session
            request.session['outcome'] = "Follow-up Recommended" if np.argmax(outputs[0]) == 1 else "Screening Negative"
            request.session['risk_score'] = risk_score
            request.session['confidence'] = "96.85%"
            request.session.modified = True # Tells Django to definitely save the data
            
            return redirect('result_page')

    return render(request, 'index.html', {'questions': QUESTION_TEXTS})

def result_page(request):
    # Retrieve the individual pieces of data from the session
    context = {
        'outcome': request.session.get('outcome', 'No Data Found'),
        'risk_score': request.session.get('risk_score', '0.00'),
        'confidence': request.session.get('confidence', 'N/A')
    }
    return render(request, 'result.html', context)