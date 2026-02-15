import os
import re
import requests
import pandas as pd
import pandas_ta as ta
from google import genai
from google.genai import types
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, make_response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Cargar variables de entorno (API Keys)
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Asegurarse de que el folder de uploads existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurar APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

if not GEMINI_API_KEY:
    raise ValueError("ERROR: Falta GEMINI_API_KEY.")

client = genai.Client(api_key=GEMINI_API_KEY)

# --- PROMPTS ---
IMAGE_PROMPT = """
Eres un analista senior de trading. Analiza la IMAGEN y responde con este formato exacto:

ANALISIS: [Detalles técnicos del gráfico]
DECISION: [OPERAR / ESPERAR / NO OPERAR]
TIPO: [COMPRA / VENTA / N/A]
RIESGO: [BAJO / MEDIO / ALTO]
MOTIVO: [Justificación de la decisión y qué invalidaría la operación]

REGLA: Sé conservador y prioriza la protección del capital.
"""

DATA_PROMPT = """
Eres un estratega de trading cuantitativo. Analiza los siguientes DATOS TÉCNICOS y responde con este formato exacto:

ANALISIS: [Interpretación de la acción del precio y los indicadores]
DECISION: [OPERAR / ESPERAR / NO OPERAR]
TIPO: [COMPRA / VENTA / N/A]
RIESGO: [BAJO / MEDIO / ALTO]
MOTIVO: [Confluencia detectada, configuración del setup y niveles clave de invalidación]

REGLA: Si los indicadores no muestran una confluencia clara, la decisión debe ser ESPERAR.
"""

# --- LÓGICA DE MERCADO (ALPHA VANTAGE) ---
class MarketService:
    BASE_URL = "https://www.alphavantage.co/query"

    @staticmethod
    def get_analysis_data(symbol_key):
        """Obtiene datos de Alpha Vantage y calcula indicadores."""
        if not ALPHA_VANTAGE_KEY:
            print("ERROR: Falta ALPHA_VANTAGE_KEY en .env")
            return None

        # Configuración Dinámica
        params = {
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "compact"
        }

        # Detectar si es Crypto o Forex
        if "BTC" in symbol_key or "ETH" in symbol_key:
            params["function"] = "CRYPTO_INTRADAY"
            params["symbol"] = symbol_key.replace("USD", "")
            params["market"] = "USD"
            params["interval"] = "5min"
        else:
            # Asumimos Forex para todo lo demás (EURUSD, GBPUSD, etc)
            params["function"] = "FX_INTRADAY"
            params["from_symbol"] = symbol_key[:3] # Ej: EUR
            params["to_symbol"] = symbol_key[3:]   # Ej: USD
            params["interval"] = "5min"

        try:
            response = requests.get(MarketService.BASE_URL, params=params)
            data = response.json()
            
            # Detectar la clave correcta de la serie temporal
            ts_key = next((k for k in data.keys() if "Time Series" in k or "Intraday" in k), None)
            
            if not ts_key:
                print(f"Error API AV: {data.get('Note', data.get('Error Message', 'Unknown Error'))}")
                return None
            
            # Convertir JSON a DataFrame
            df = pd.DataFrame.from_dict(data[ts_key], orient='index')
            df = df.astype(float)
            
            # Renombrar columnas para pandas_ta
            df.columns = [c.split('. ')[1].capitalize() for c in df.columns] 
            df = df.sort_index()

            # Calcular Indicadores
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)

            last_row = df.iloc[-1]

            data_summary = {
                'symbol': symbol_key,
                'price': round(float(last_row['Close']), 5),
                'rsi': round(float(last_row['RSI']), 2) if pd.notna(last_row['RSI']) else 0,
                'ema20': round(float(last_row['EMA_20']), 5) if pd.notna(last_row['EMA_20']) else 0,
                'ema50': round(float(last_row['EMA_50']), 5) if pd.notna(last_row['EMA_50']) else 0,
                'trend': 'ALCISTA' if last_row['EMA_20'] > last_row['EMA_50'] else 'BAJISTA',
                'change_pct': round(((float(last_row['Close']) - float(df.iloc[-2]['Close'])) / float(df.iloc[-2]['Close'])) * 100, 3) 
            }
            return data_summary

        except Exception as e:
            print(f"Excepción en MarketService: {e}")
            return None

def analyze_data_with_ai(data_summary):
    """Envia el resumen de indicadores a Gemini."""
    input_text = f"""
    Datos de Mercado EN VIVO ({data_summary['symbol']}):
    - Precio: {data_summary['price']}
    - RSI (14): {data_summary['rsi']}
    - EMA 20: {data_summary['ema20']} - EMA 50: {data_summary['ema50']}
    - Tendencia Técnica: {data_summary['trend']}
    - Momentum: {data_summary['change_pct']}%
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[data_summary['symbol'], input_text, DATA_PROMPT]
        )
        return response.text
    except Exception as e:
        return f"ERROR_IA: {str(e)}"

def analyze_chart(file_path, prompt):
    """Envía la imagen a Gemini."""
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt]
        )
        return response.text
    except Exception as e:
        return f"ERROR_IA: {str(e)}"

def parse_analysis(text):
    """Parser robusto para la respuesta de la IA."""
    res = {'tecnico': '...', 'decision': 'ESPERAR', 'tipo': 'N/A', 'riesgo': 'ALTO', 'motivo': '...'}
    clean = text.replace('**', '').strip()
    
    mapeo = {
        'tecnico': ['ANALISIS', 'ESTRUCTURA'],
        'decision': ['DECISION', 'DECISIÓN'],
        'tipo': ['TIPO', 'OPERACION'],
        'riesgo': ['RIESGO'],
        'motivo': ['MOTIVO', 'JUSTIFICACION']
    }
    
    for line in clean.split('\n'):
        line = line.strip()
        if not line: continue
        upper = line.upper()
        
        for key, keywords in mapeo.items():
            for kw in keywords:
                if upper.startswith(kw):
                    parts = line.split(':', 1)
                    if len(parts) > 1: res[key] = parts[1].strip()
    
    return res

@app.route('/manifest.json')
def manifest(): return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def service_worker():
    response = make_response(send_from_directory('static', 'sw.js'))
    response.headers['Content-Type'] = 'application/javascript'
    return response

@app.route('/uploads/<filename>')
def uploaded_file(filename): return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    mode = 'image'
    
    # Lista ampliada de Pares Forex y Crypto
    symbols = {
        'EUR/USD - Euro Dólar': 'EURUSD',
        'GBP/USD - Libra Dólar': 'GBPUSD',
        'USD/JPY - Dólar Yen': 'USDJPY',
        'USD/CHF - Dólar Franco': 'USDCHF',
        'AUD/USD - Australiano': 'AUDUSD',
        'USD/CAD - Canadiense': 'USDCAD',
        'XAU/USD - Oro Spot': 'XAUUSD',
        'BTC/USD - Bitcoin': 'BTCUSD',
        'ETH/USD - Ethereum': 'ETHUSD'
    }

    if request.method == 'POST':
        mode = request.form.get('mode', 'image')
        
        if mode == 'image':
            file = request.files.get('chart')
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                result = parse_analysis(analyze_chart(path, IMAGE_PROMPT))
                result['filename'] = filename
        
        elif mode == 'api':
            target = request.form.get('symbol', 'EURUSD')
            data = MarketService.get_analysis_data(target)
            if data:
                result = parse_analysis(analyze_data_with_ai(data))
                result['market_data'] = data
            else:
                result = {'error': "Error de Conexión Alpha Vantage (Verifica tu API KEY o límites de uso)."}
                
    return render_template('index.html', result=result, mode=mode, symbols=symbols)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
