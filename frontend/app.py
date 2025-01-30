from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
from models.retriever import rag_chatbot
from pdf_processing.pdf_processing_pipeline import PDFProcessingPipeline

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session-based flash messages

# Upload folder for storing PDF files
UPLOAD_FOLDER = 'data/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'pdf_files' not in request.files:
        flash('No files part', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('pdf_files')
    pdf_paths = []

    # Save each file to the server
    for file in files:
        if file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            pdf_paths.append(file_path)
        else:
            flash(f"File '{file.filename}' is not a valid PDF!", 'error')
            return redirect(url_for('index'))

    # Process PDFs and push data to DB
    pipeline = PDFProcessingPipeline(pdf_paths=pdf_paths, push_to_db=True)
    pipeline.process_pdfs()

    flash('Files uploaded and processed successfully!', 'success')
    
    # Redirect to the chat page after processing
    return redirect(url_for('chat'))

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    results = rag_chatbot(query)
    
    # Return only the answer, file name, and chunk number
    simplified_results = [{"answer": res["answer"], "file": res["file"], "chunk_number": res["chunk_number"]} for res in results]
    
    return jsonify(simplified_results)

if __name__ == "__main__":
    app.run(debug=True)
