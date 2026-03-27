import time
import io
import csv
from flask import Flask, render_template, request, redirect, url_for

# Extracted decoupled configuration and native capabilities bounds
from config import Config
from services.nlp_service import NLPService
from services.ml_service import MLService
from services.llm_service import LLMService

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Core Services into the persistent operational instance memory
nlp_service = NLPService()
ml_service = MLService(Config.MODEL_PATH, Config.VEC_PATH)
llm_service = LLMService(Config.METADATA_PATH, Config.GEMINI_API_KEY)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyzer')
def analyzer_route():
    # Pass uniquely scraped variables via the decoupled LLM configuration map
    return render_template('analyzer.html', apps=llm_service.unique_apps)


@app.route('/predict', methods=['POST'])
def predict():
    if not ml_service.is_trained:
        return "Internal Error: ML Models are not trained or missing. Run model_builder.py first.", 500

    single_app_name = request.form.get('app_name', 'Unknown App').strip()
    raw_review = request.form.get('review', '').strip()
    file = request.files.get('file')

    # ==========================
    # BATCH PAYLOAD PROCESSING
    # ==========================
    if file and file.filename.endswith('.csv'):
        try:
            stream = io.StringIO(file.stream.read().decode("UTF8", errors="ignore"), newline=None)
            csv_input = csv.reader(stream)
            headers = next(csv_input)
            
            text_col_idx, app_col_idx = -1, -1
            
            for i, item in enumerate(headers):
                val = item.lower().strip()
                if val in ['review', 'content', 'text', 'translated_review']: text_col_idx = i
                elif val in ['app', 'app_name', 'app name']: app_col_idx = i
                    
            if text_col_idx == -1:
                text_col_idx = 1 if len(headers) > 1 else 0
                
            results_list = []
            negative_feedback_by_app = {}
            unique_payload_apps = set()
            stats = {'pos': 0, 'neu': 0, 'neg': 0, 'total_conf': 0, 'critical_alerts': 0}
            
            for row in csv_input:
                if len(row) > text_col_idx:
                    text_val = row[text_col_idx].strip()
                    app_val = row[app_col_idx].strip() if (app_col_idx != -1 and len(row) > app_col_idx) else single_app_name
                    
                    if text_val:
                        # Pipe text strings sequentially through encapsulated backend capabilities 
                        cleaned = nlp_service.preprocess_text(text_val)
                        res = ml_service.analyze_payload(cleaned, text_val, app_val)
                        
                        results_list.append(res)
                        unique_payload_apps.add(app_val)
                        
                        if res['sentiment'] == 'Positive': 
                            stats['pos'] += 1
                        elif res['sentiment'] == 'Negative': 
                            stats['neg'] += 1
                            stats['critical_alerts'] += 1
                            if app_val not in negative_feedback_by_app: 
                                negative_feedback_by_app[app_val] = []
                            negative_feedback_by_app[app_val].append(text_val)
                        else: 
                            stats['neu'] += 1
                        
                        stats['total_conf'] += res['confidence']
                        
            stats['avg_conf'] = round(stats['total_conf'] / len(results_list), 1) if len(results_list) > 0 else 0
                
            # SEGMENTED API CALLS via LLM Service Integrations 
            ai_summaries = {}
            for app_id in unique_payload_apps:
                if app_id in negative_feedback_by_app and negative_feedback_by_app[app_id]:
                    summary = llm_service.get_executive_summary(app_id, negative_feedback_by_app[app_id])
                    ai_summaries[app_id] = summary
                    # Enforce strict throttling sequences per-app against the external provider endpoints
                    time.sleep(1.0)
                else:
                    ai_summaries[app_id] = "No critical alerts found. The current trajectory is stable."
                
            return render_template('results.html', results=results_list, ai_summaries=ai_summaries, is_batch=True, stats=stats)
            
        except Exception as e:
            return f"Error executing parsing metrics over uploaded CSV dataset: {e}", 400

    # ==========================
    # SINGLE RECORD PROCESSING
    # ==========================
    elif raw_review:
        cleaned = nlp_service.preprocess_text(raw_review)
        res = ml_service.analyze_payload(cleaned, raw_review, single_app_name)
        
        summary = llm_service.get_executive_summary(single_app_name, [raw_review]) if res['sentiment'] == 'Negative' else "No critical alerts triggered. Trajectory is stable."
        ai_summaries = { single_app_name: summary }
        
        return render_template('results.html', results=[res], ai_summaries=ai_summaries, is_batch=False)
        
    else:
        return redirect(url_for('analyzer_route'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
