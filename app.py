from flask import Flask, render_template, request, jsonify, send_file, g
from werkzeug.utils import secure_filename
from tools import file_to_desired_dict, generate_example_sentence
import json
import os

app = Flask(__name__)

# 确保上传文件夹存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 使用简单的内存存储来保存单词数据
words_data_storage = {}

@app.before_request
def before_request():
    g.words_data = words_data_storage.get('data', [])

@app.after_request
def after_request(response):
    words_data_storage['data'] = g.words_data
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            if filename.lower().endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    g.words_data = json.load(f)
            else:
                g.words_data = file_to_desired_dict(file_path)
            return jsonify(g.words_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
@app.route('/export', methods=['POST'])
def export_data():
    with open('exported_data.json', 'w', encoding='utf-8') as f:
        json.dump(g.words_data, f, ensure_ascii=False, indent=2)
    return send_file('exported_data.json', as_attachment=True)

@app.route('/generate_example', methods=['POST'])
def generate_example():
    data = request.json
    word = data['word']
    selected_content = data['selected_content']
    
    try:
        example = generate_example_sentence(word, selected_content)
        
        # 更新单词数据
        for word_data in g.words_data:
            if word_data['word'] == word:
                if 'examples' not in word_data:
                    word_data['examples'] = []
                word_data['examples'].append(example)
                break
        
        return jsonify({'example': example})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)