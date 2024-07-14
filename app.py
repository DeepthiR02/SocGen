from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename
from vba_analyser import generate_documentation
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith(('.xlsm', '.xls', '.xlsx')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('process_file', filename=filename))
    return render_template('index.html')

@app.route('/process/<filename>')
@cache.memoize(timeout=300)  # Cache results for 5 minutes
def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    documentation, flowchart_path = generate_documentation(file_path)
    return render_template('result.html', documentation=documentation, flowchart_path=flowchart_path)

@app.route('/flowchart/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)