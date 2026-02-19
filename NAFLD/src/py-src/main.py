# pip freeze the erqs from the backend and just integrate it here
# add new gitignores here as well from the other project
# 


# python -m pip install -r ./setup.txt
# flask --app .\NAFLD\src\py-src\main.py run
import os
from flask import Flask,jsonify,send_file, request, Response
from datetime import datetime
from flask_cors import CORS
import sys
import zipfile
import io
import csv
from werkzeug.utils import secure_filename
from nafld import analyze_single_file, preview_single_file, analyze_single_file_patched

# sys.path.append("C:\\Projects\\Machine Learning\\NAFLD\\NAFLD-project\\NAFLD\\src\\py-src\\nafld.py")
from nafld import process_all_images
import json
import queue
import threading
app = Flask(__name__)
CORS(app, expose_headers=['Content-Disposition'])

# Allow uploads up to 2 GB (chunked uploads bypass this but regular /upload needs it)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB

UPLOAD_FOLDER = 'C:\\Users\\alexg\\Documents\\NAFLDimages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'svs'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

upload_file_dict = {}
# upload_file_list = []

# Look into restricting access from other endpoints than arent localhost?
# CORS(app, resources={r"/home": {"origins": "localhost:3000"}})

@app.route("/analyze/<filename>", methods=['GET'])
def analyze_file(filename):
    print(f"Analyzing {filename}...")

    file_path = resolve_uploaded_file_path(filename)

    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    # 2. Call the brain
    result = analyze_single_file(file_path)

    return jsonify(result), 200


def _is_patchable(file_path):
    """Return True if the file should use patch-based SSE analysis."""
    if file_path.lower().endswith('.svs'):
        return True
    if file_path.lower().endswith(('.tif', '.tiff')):
        return os.path.getsize(file_path) > 50 * 1024 * 1024
    return False


@app.route("/analyze-stream/<filename>", methods=['GET'])
def analyze_file_stream(filename):
    """SSE endpoint — streams patch progress then the final result."""
    file_path = resolve_uploaded_file_path(filename)
    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    if not _is_patchable(file_path):
        # Not a patch candidate: run normal analysis and return as a single SSE result event
        result = analyze_single_file(file_path)
        def single_event():
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
        return Response(single_event(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    progress_queue = queue.Queue()

    def on_progress(current, total, tissue_count):
        progress_queue.put({
            'type': 'progress',
            'current': current,
            'total': total,
            'tissue_patches': tissue_count,
        })

    result_holder = [None]

    def worker():
        result_holder[0] = analyze_single_file_patched(file_path, progress_callback=on_progress)
        progress_queue.put({'type': 'done'})

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    def generate():
        while True:
            try:
                msg = progress_queue.get(timeout=120)
            except queue.Empty:
                break
            if msg['type'] == 'done':
                yield f"data: {json.dumps({'type': 'result', 'data': result_holder[0]})}\n\n"
                break
            else:
                yield f"data: {json.dumps(msg)}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route("/preview/<filename>", methods=['GET'])
def preview_file(filename):
    print(f"Previewing {filename}...")

    file_path = resolve_uploaded_file_path(filename)

    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    result = preview_single_file(file_path)
    return jsonify(result), 200


@app.route("/download-single/<filename>", methods=['GET'])
def download_single_file_csv(filename):
    file_path = resolve_uploaded_file_path(filename)
    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    result = analyze_single_file(file_path)
    if result.get('status') != 'success':
        return jsonify({'error': result.get('message', 'Analysis failed')}), 500

    membership_scores = result.get('membership_scores') or {}

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['image_name', 'percentage', 'cluster_label', 'None', 'Perisinusoidal', 'Bridging', 'Cirrosis'])
    writer.writerow([
        os.path.basename(file_path),
        result.get('fibrosis_ratio', ''),
        result.get('cluster_label', ''),
        membership_scores.get('None', result.get('None', '')),
        membership_scores.get('Perisinusoidal', result.get('Perisinusoidal', '')),
        membership_scores.get('Bridging', result.get('Bridging', '')),
        membership_scores.get('Cirrosis', result.get('Cirrosis', '')),
    ])

    download_name = f"result_{os.path.basename(file_path)}.csv"
    response = Response(csv_buffer.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename="{download_name}"'
    return response


def resolve_uploaded_file_path(filename):

    # 1. Find the file
    # Check if it's in the dict (from your upload logic) or just in the folder
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path) and filename in upload_file_dict:
        # Logic for folders/zips if you kept that structure
        folder = upload_file_dict[filename]
        file_path = os.path.join(folder, filename)
    elif not os.path.exists(file_path):
        # Fallback to simple lookup in UPLOAD_FOLDER
        # Note: Your upload adds a timestamp like _34:12 to filenames. 
        # For simplicity in testing, we might need to exact match.
        # Let's assume for now you are testing with a file you just put there manually 
        # or the upload logic is simplified.
        for f in os.listdir(UPLOAD_FOLDER):
            if f.startswith(filename):
                file_path = os.path.join(UPLOAD_FOLDER, f)
                break

    if not os.path.exists(file_path):
        return None

    return file_path

@app.route("/home")
def home():
    return {
  "blogs": [
    {
      "title": "My First Blog",
      "body": "Why do we use it?\nIt is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).\n\n\nWhere does it come from?\nContrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of \"de Finibus Bonorum et Malorum\" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, \"Lorem ipsum dolor sit amet..\", comes from a line in section 1.10.32.\n\nThe standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from \"de Finibus Bonorum et Malorum\" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham.\n\nWhere can I get some?\nThere are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc.",
      "author": "mario",
      "id": 1
    },
    {
      "title": "Opening Party!",
      "body": "Why do we use it?\nIt is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).\n\n\nWhere does it come from?\nContrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of \"de Finibus Bonorum et Malorum\" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, \"Lorem ipsum dolor sit amet..\", comes from a line in section 1.10.32.\n\nThe standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from \"de Finibus Bonorum et Malorum\" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham.\n\nWhere can I get some?\nThere are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc.",
      "author": "yoshi",
      "id": 2
    }
  ]
}


@app.route("/upload", methods =['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    print(f'received: {file}')
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}'}), 400

    original_filename = secure_filename(file.filename)
    root, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    safe_filename = f"{root}_{timestamp}{ext.lower()}"

    save_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(save_path)

    return jsonify({
        'message': 'File successfully uploaded',
        'filename': safe_filename
    }), 200

# Dict to track chunked uploads: maps original filename -> timestamped safe name
_chunked_upload_names = {}

@app.route("/largefile", methods=['POST'])
def upload_largefile():
    chunk = request.files['file']
    resumable_filename = request.form['resumableFilename']
    resumable_chunk_number = int(request.form['resumableChunkNumber'])
    total_chunks = int(request.form['resumableTotalChunks'])

    # On first chunk, generate a timestamped safe name (same logic as /upload)
    if resumable_chunk_number == 1 and resumable_filename not in _chunked_upload_names:
        original = secure_filename(resumable_filename)
        root, ext = os.path.splitext(original)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        _chunked_upload_names[resumable_filename] = f"{root}_{timestamp}{ext.lower()}"

    safe_name = _chunked_upload_names.get(resumable_filename, secure_filename(resumable_filename))
    full_file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(full_file_path, 'ab') as f:
            f.write(chunk.read())
    except Exception as e:
        print(f"Chunk write error: {e}")
        return jsonify({"status": "Error writing chunk to file"}), 400

    print(f"Chunk {resumable_chunk_number}/{total_chunks} for {safe_name}")

    if resumable_chunk_number == total_chunks:
        # Clean up tracking dict
        _chunked_upload_names.pop(resumable_filename, None)
        return jsonify({
            "status": "File upload complete",
            "filename": safe_name
        }), 200

    return jsonify({"status": "Chunk upload successful"}), 200



@app.route("/fake", methods = ['POST'])
def upload_json():
    data = request.get_json()
    print(data)  # Process your data here
    return jsonify({"message": "Data received!", "data": data}), 200
    


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(folder_path):
    #check if zip & extract
    df = process_all_images(folder_path)

    # send back as csv?

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    if(filename not in upload_file_dict):
        return jsonify({'error': 'File not found'}), 404
    
    # Extract Any zip files
    folder_path = upload_file_dict[filename]
    for filename in os.listdir(folder_path):
      if filename.endswith('.zip'):  # Check if the file has a .zip extension
          zip_file_path = os.path.join(folder_path, filename)
          
          # Check if the file is a valid zip file
          if zipfile.is_zipfile(zip_file_path):
              print(f"Found ZIP file: {filename}")
              
              # Extract the zip file in the same directory
              with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                  zip_ref.extractall(folder_path)
              print(f"Extracted {filename} to {folder_path}")
          else:
              print(f"{filename} is not a valid zip file.")

    csv_file_name = 'output.csv'
    csv_path = os.path.join(folder_path,csv_file_name)
    df = process_all_images(folder_path)
    df.to_csv(csv_path, index=False)

    try:
        response = send_file(csv_path, as_attachment=True)
        response.headers["Content-Disposition"] = f"attachment; filename={csv_file_name}"
        return send_file(csv_path, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


    

@app.route("/fullFileUpload", methods =['POST'])
def full_file_upload():
    chunk = request.files['file'] 
    resumable_filename = request.form['resumableFilename']  

    if resumable_filename not in upload_file_dict:
      
      current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
      new_folder_path = os.path.join(UPLOAD_FOLDER, current_time)
      upload_file_dict[resumable_filename] = new_folder_path
      os.makedirs(new_folder_path, exist_ok=True)



    resumable_chunk_number = request.form['resumableChunkNumber']  # Chunk index (1-based)
    total_chunks = int(request.form['resumableTotalChunks'])
    full_file_path = os.path.join(upload_file_dict[resumable_filename], resumable_filename)

    print(full_file_path)
    try:
        with open(full_file_path,'ab') as chunked_file:
            chunked_file.write(chunk.read())
    except Exception:
        return jsonify({"status": "Error writing chunk to file"}), 400
    
    if(int(resumable_chunk_number) == total_chunks):
        print("Returned file")
        # return jsonify({"status": {'file_status':"file upload complete","fileName":f"{resumable_filename}"}}), 200
        return jsonify({"status": "file upload complete","fileName":f"{resumable_filename}"}), 200
    
    # print(f"res num: {resumable_chunk_number} total: {total_chunks}")
    return jsonify({"status": "Chunk upload successful"}), 200
    


if __name__ == "__main__":
    app.run(debug=True)

