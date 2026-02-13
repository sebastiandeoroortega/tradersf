import os
import re
from google import genai
from google.genai import types
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, make_response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Cargar variables de entorno (API Key)
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Asegurarse de que el folder de uploads existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurar la API de Google Gemini (Nuevo SDK)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Esto ayuda a detectar el error en Railway/Render
    raise ValueError("ERROR: No se encontró la variable GEMINI_API_KEY. Configúrala en el panel de control de Railway.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Prompt ultra-estricto y redundante
SYSTEM_PROMPT = """
Eres un analista de trading. Analiza la imagen y responde con este formato exacto:

ANALISIS: [Detalles técnicos aquí]
DECISION: [OPERAR / ESPERAR / NO OPERAR]
TIPO: [COMPRA / VENTA / N/A]
RIESGO: [BAJO / MEDIO / ALTO]
MOTIVO: [Explicación de por qué tomas esa decisión]

REGLA: Usa solo estas palabras clave y sé muy claro.
"""

def analyze_chart(file_path):
    """Envía la imagen a Gemini."""
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                SYSTEM_PROMPT
            ]
        )
        print("--- RESPUESTA IA ---")
        print(response.text)
        print("--------------------")
        return response.text
    except Exception as e:
        return f"ERROR_IA: {str(e)}"

def parse_analysis(text):
    """Extrae la información de forma ultra-flexible."""
    res = {
        'tecnico': 'No detectado',
        'decision': 'ESPERAR',
        'tipo': 'N/A',
        'riesgo': 'ALTO',
        'motivo': 'No detectado'
    }
    
    clean = text.replace('**', '').replace('###', '').strip()
    
    # Mapeo de búsqueda por palabras clave (sin tildes y flexible)
    mapeo = {
        'tecnico': ['ANALISIS', 'ESTRUCTURA', 'DESCRIP'],
        'decision': ['DECISION', 'DECISIÓN', 'RECOMEND'],
        'tipo': ['TIPO', 'OPERACIO', 'ACCIO', 'POSICIO'],
        'riesgo': ['RIESGO', 'RISK'],
        'motivo': ['MOTIVO', 'MOTIVA', 'JUSTIFICA', 'RAZON', 'POR QUE']
    }
    
    lineas = clean.split('\n')
    seccion_actual = None
    
    for l in lineas:
        l_upper = l.upper().strip()
        if not l_upper: continue
        
        encontrado = False
        for key, keywords in mapeo.items():
            for kw in keywords:
                if l_upper.startswith(kw):
                    seccion_actual = key
                    # Extraer lo que hay tras los dos puntos
                    partes = l.split(':', 1)
                    if len(partes) > 1:
                        res[key] = partes[1].strip()
                    else:
                        res[key] = ""
                    encontrado = True
                    break
            if encontrado: break
            
        if not encontrado and seccion_actual:
            if res[seccion_actual] == 'No detectado' or res[seccion_actual] == "":
                res[seccion_actual] = l.strip()
            else:
                res[seccion_actual] += " " + l.strip()

    # Normalización de la decisión
    d = res['decision'].upper()
    if 'OPERAR' in d and 'NO' not in d: res['decision'] = 'OPERAR'
    elif 'ESPERAR' in d: res['decision'] = 'ESPERAR'
    else: res['decision'] = 'NO OPERAR'
    
    return res

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def service_worker():
    response = make_response(send_from_directory('static', 'sw.js'))
    response.headers['Content-Type'] = 'application/javascript'
    return response

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('chart')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            raw = analyze_chart(path)
            if "ERROR_IA:" in raw:
                result = {'error': raw}
            else:
                result = parse_analysis(raw)
                result['filename'] = filename
                
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
