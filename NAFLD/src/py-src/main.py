# pip freeze the erqs from the backend and just integrate it here
# add new gitignores here as well from the other project

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
from nafld import analyze_single_file, preview_single_file, analyze_single_file_patched, rethreshold, rethreshold_area, reset_area, undo_area, get_delta_map, get_excluded_mask, pil_to_b64, classify_from_mask, analyze_area, classify_area

# sys.path.append("C:\\Projects\\Machine Learning\\NAFLD\\NAFLD-project\\NAFLD\\src\\py-src\\nafld.py")
from nafld import process_all_images
import json
import queue
import threading

from auth import auth_bp, init_db, login_required

# Serve the React build in production when NAFLD_STATIC_DIR is set
_static_dir = os.environ.get('NAFLD_STATIC_DIR')
if _static_dir:
    app = Flask(__name__, static_folder=_static_dir, static_url_path='')
else:
    app = Flask(__name__)
CORS(app, expose_headers=['Content-Disposition'],
     allow_headers=['Content-Type', 'Authorization'])

# Initialise user database on startup
init_db()

# Register auth blueprint (provides /login, /refresh, /me)
app.register_blueprint(auth_bp)

# Allow uploads up to 2 GB (chunked uploads bypass this but regular /upload needs it)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB

UPLOAD_FOLDER = os.environ.get('NAFLD_UPLOAD_FOLDER', 'C:\\Users\\alexg\\Documents\\NAFLDimages')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'svs'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

upload_file_dict = {}
# upload_file_list = []

# Cache analysis results so CSV download doesn't re-run the full pipeline.
# Keyed by resolved file path → result dict.
_analysis_cache = {}

# Disk-backed metadata directory for CSV downloads (survives across workers/restarts)
_META_DIR = os.path.join(UPLOAD_FOLDER, '.meta')
os.makedirs(_META_DIR, exist_ok=True)

def _save_result_meta(filename, result):
    """Persist the CSV-relevant fields to a small JSON sidecar on disk."""
    meta = {
        'status': result.get('status'),
        'fibrosis_ratio': result.get('fibrosis_ratio'),
        'membership_scores': result.get('membership_scores'),
    }
    path = os.path.join(_META_DIR, f"{filename}.json")
    with open(path, 'w') as f:
        json.dump(meta, f)

def _load_result_meta(filename):
    """Load CSV fields from disk sidecar. Returns dict or None."""
    path = os.path.join(_META_DIR, f"{filename}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# Look into restricting access from other endpoints than arent localhost?
# CORS(app, resources={r"/home": {"origins": "localhost:3000"}})

@app.route("/analyze/<filename>", methods=['GET'])
@login_required
def analyze_file(filename):
    print(f"Analyzing {filename}...")

    file_path = resolve_uploaded_file_path(filename)

    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    # 2. Call the brain
    result = analyze_single_file(file_path)

    # Cache the result so CSV download can reuse it
    _analysis_cache[file_path] = result
    _save_result_meta(filename, result)

    return jsonify(result), 200


def _is_patchable(file_path):
    """Return True if the file should use patch-based SSE analysis."""
    if file_path.lower().endswith('.svs'):
        return True
    if file_path.lower().endswith(('.tif', '.tiff')):
        return os.path.getsize(file_path) > 50 * 1024 * 1024
    return False


@app.route("/analyze-stream/<filename>", methods=['GET'])
@login_required
def analyze_file_stream(filename):
    """SSE endpoint — streams patch progress then the final result."""
    file_path = resolve_uploaded_file_path(filename)
    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    if not _is_patchable(file_path):
        # Not a patch candidate: run normal analysis and return as a single SSE result event
        result = analyze_single_file(file_path)
        _analysis_cache[file_path] = result
        _save_result_meta(filename, result)
        def single_event():
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
        return Response(single_event(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    progress_queue = queue.Queue()

    def on_progress(current, total, tissue_count, **kwargs):
        msg = {
            'type': 'progress',
            'current': current,
            'total': total,
            'tissue_patches': tissue_count,
        }
        msg.update(kwargs)
        progress_queue.put(msg)

    result_holder = [None]

    def worker():
        result_holder[0] = analyze_single_file_patched(file_path, progress_callback=on_progress)
        _analysis_cache[file_path] = result_holder[0]
        _save_result_meta(filename, result_holder[0])
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
@login_required
def preview_file(filename):
    print(f"Previewing {filename}...")

    file_path = resolve_uploaded_file_path(filename)

    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    result = preview_single_file(file_path)
    return jsonify(result), 200


@app.route("/rethreshold/<filename>", methods=['GET'])
@login_required
def rethreshold_file(filename):
    """Apply a user-chosen threshold to cached deconvolution data."""
    thresh_str = request.args.get('threshold')
    if thresh_str is None:
        return jsonify({'error': 'Missing threshold parameter'}), 400
    try:
        new_thresh = float(thresh_str)
    except ValueError:
        return jsonify({'error': 'Invalid threshold value'}), 400

    # The cache key is the basename used during analysis
    result = rethreshold(filename, new_thresh)
    if result is None:
        return jsonify({'error': 'No cached data for this file. Re-run analysis first.'}), 404

    mask_pil, total_px, ratio, tissue_count = result
    return jsonify({
        'status': 'success',
        'filtered_image': f"data:image/jpeg;base64,{pil_to_b64(mask_pil)}",
        'fibrosis_ratio': float(ratio),
        'threshold': new_thresh,
    }), 200


@app.route("/rethreshold-area/<filename>", methods=['GET'])
@login_required
def rethreshold_area_file(filename):
    """Apply a relative threshold delta to a specific region."""
    try:
        delta = float(request.args.get('delta', ''))
        x1 = float(request.args.get('x1', ''))
        y1 = float(request.args.get('y1', ''))
        x2 = float(request.args.get('x2', ''))
        y2 = float(request.args.get('y2', ''))
    except (ValueError, TypeError):
        return jsonify({'error': 'Missing or invalid parameters (delta, x1, y1, x2, y2)'}), 400

    result = rethreshold_area(filename, delta, x1, y1, x2, y2)
    if result is None:
        return jsonify({'error': 'No cached data or invalid region'}), 404

    mask_pil, total_px, ratio, tissue_count = result
    delta_map_b64 = get_delta_map(filename)
    return jsonify({
        'status': 'success',
        'filtered_image': f"data:image/jpeg;base64,{pil_to_b64(mask_pil)}",
        'fibrosis_ratio': float(ratio),
        'delta_map': f"data:image/png;base64,{delta_map_b64}" if delta_map_b64 else None,
        'has_local_edits': True,
    }), 200


@app.route("/reset-area/<filename>", methods=['GET'])
@login_required
def reset_area_file(filename):
    """Reset threshold delta to zero in a specific region."""
    try:
        x1 = float(request.args.get('x1', ''))
        y1 = float(request.args.get('y1', ''))
        x2 = float(request.args.get('x2', ''))
        y2 = float(request.args.get('y2', ''))
    except (ValueError, TypeError):
        return jsonify({'error': 'Missing or invalid parameters (x1, y1, x2, y2)'}), 400

    result = reset_area(filename, x1, y1, x2, y2)
    if result is None:
        return jsonify({'error': 'No cached data or invalid region'}), 404

    mask_pil, total_px, ratio, tissue_count = result
    delta_map_b64 = get_delta_map(filename)
    from nafld import _deconv_cache
    has_edits = _deconv_cache.get(filename, {}).get('has_local_edits', False)
    return jsonify({
        'status': 'success',
        'filtered_image': f"data:image/jpeg;base64,{pil_to_b64(mask_pil)}",
        'fibrosis_ratio': float(ratio),
        'delta_map': f"data:image/png;base64,{delta_map_b64}" if delta_map_b64 else None,
        'has_local_edits': has_edits,
    }), 200


@app.route("/undo-area/<filename>", methods=['GET'])
@login_required
def undo_area_file(filename):
    """Undo the last area modification."""
    result = undo_area(filename)
    if result is None:
        return jsonify({'error': 'Nothing to undo'}), 404

    mask_pil, total_px, ratio, tissue_count = result
    delta_map_b64 = get_delta_map(filename)
    from nafld import _deconv_cache
    has_edits = _deconv_cache.get(filename, {}).get('has_local_edits', False)
    return jsonify({
        'status': 'success',
        'filtered_image': f"data:image/jpeg;base64,{pil_to_b64(mask_pil)}",
        'fibrosis_ratio': float(ratio),
        'delta_map': f"data:image/png;base64,{delta_map_b64}" if delta_map_b64 else None,
        'has_local_edits': has_edits,
    }), 200


@app.route("/excluded-mask/<filename>", methods=['GET'])
@app.route("/preview-excluded/<filename>", methods=['GET'])
@login_required
def excluded_mask_file(filename):
    """Return a base64 RGBA PNG that paints excluded (non-tissue) pixels
    in green. These are the exact pixels removed from the extent denominator."""
    overlay_b64 = get_excluded_mask(filename)
    if overlay_b64 is None:
        return jsonify({'error': 'No cached data for this file. Re-run analysis first.'}), 404
    return jsonify({
        'status': 'success',
        'overlay': f"data:image/png;base64,{overlay_b64}",
    }), 200


def _parse_region_args():
    """Parse normalised x1,y1,x2,y2 query params. Returns tuple or None on error."""
    try:
        return (float(request.args.get('x1', '')),
                float(request.args.get('y1', '')),
                float(request.args.get('x2', '')),
                float(request.args.get('y2', '')))
    except (ValueError, TypeError):
        return None


@app.route("/analyze-area/<filename>", methods=['GET'])
@login_required
def analyze_area_file(filename):
    """Return fibrosis extent for a normalised region under the magnifier."""
    region = _parse_region_args()
    if region is None:
        return jsonify({'error': 'Missing or invalid x1/y1/x2/y2'}), 400
    result = analyze_area(filename, *region)
    if result is None:
        return jsonify({'error': 'No cached data or invalid region'}), 404
    result['status'] = 'success'
    return jsonify(result), 200


@app.route("/classify-area/<filename>", methods=['GET'])
@app.route("/classify-mask-area/<filename>", methods=['GET'])
@login_required
def classify_area_file(filename):
    """Run VGG16+PCA+FCM on the magnifier region's mask. Returns membership scores."""
    region = _parse_region_args()
    if region is None:
        return jsonify({'error': 'Missing or invalid x1/y1/x2/y2'}), 400
    result = classify_area(filename, *region)
    if result is None:
        return jsonify({'error': 'No cached data or invalid region'}), 404
    return jsonify(result), 200


@app.route("/classify-mask/<filename>", methods=['GET'])
@login_required
def classify_mask_file(filename):
    """SSE endpoint — streams patch scanning progress then the final classification result."""
    from nafld import _deconv_cache
    entry = _deconv_cache.get(filename)
    if entry is None:
        return jsonify({'error': 'No cached analysis data. Analyze the image first.'}), 404

    progress_queue = queue.Queue()

    def on_progress(current, total, tissue_count, **kwargs):
        msg = {
            'type': 'progress',
            'current': current,
            'total': total,
            'tissue_patches': tissue_count,
        }
        msg.update(kwargs)
        progress_queue.put(msg)

    result_holder = [None]

    def worker():
        result_holder[0] = classify_from_mask(filename, progress_callback=on_progress)
        progress_queue.put({'type': 'done'})

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    def generate():
        while True:
            try:
                msg = progress_queue.get(timeout=300)
            except queue.Empty:
                break
            if msg['type'] == 'done':
                yield f"data: {json.dumps({'type': 'result', 'data': result_holder[0]})}\n\n"
                break
            else:
                yield f"data: {json.dumps(msg)}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route("/download-single/<filename>", methods=['GET'])
@login_required
def download_single_file_csv(filename):
    # Try sources in order: in-memory cache → disk sidecar → re-analyze
    result = None
    for cached_path, cached_result in _analysis_cache.items():
        if os.path.basename(cached_path) == filename or cached_path.endswith(filename):
            result = cached_result
            break

    if result is None:
        result = _load_result_meta(filename)

    if result is None:
        file_path = resolve_uploaded_file_path(filename)
        if not file_path:
            return jsonify({'error': 'File not found and no cached analysis available'}), 404
        result = analyze_single_file(file_path)
        _analysis_cache[file_path] = result
        _save_result_meta(filename, result)

    if result.get('status') != 'success':
        return jsonify({'error': result.get('message', 'Analysis failed')}), 500

    # If the user adjusted the threshold, use that ratio in the CSV
    override_ratio = request.args.get('fibrosis_ratio')
    fibrosis_ratio = float(override_ratio) if override_ratio is not None else result.get('fibrosis_ratio', '')

    # Prefer classification result from Diagnose workflow if available
    classify_scores = request.args.get('classify_scores')
    if classify_scores:
        try:
            membership_scores = json.loads(classify_scores)
        except (json.JSONDecodeError, TypeError):
            membership_scores = result.get('membership_scores') or {}
    else:
        membership_scores = result.get('membership_scores') or {}

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['image_name', 'extent_percentage', 'None', 'Perisinusoidal', 'Bridging', 'Cirrosis'])
    writer.writerow([
        filename,
        fibrosis_ratio,
        membership_scores.get('None', ''),
        membership_scores.get('Perisinusoidal', ''),
        membership_scores.get('Bridging', ''),
        membership_scores.get('Cirrosis', ''),
    ])

    download_name = f"result_{filename}.csv"
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
@login_required
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
@login_required
def upload_largefile():
    chunk = request.files['file']
    resumable_filename = request.form['resumableFilename']
    resumable_chunk_number = int(request.form['resumableChunkNumber'])
    total_chunks = int(request.form['resumableTotalChunks'])

    # Build a deterministic safe name from the original filename + total chunks
    # so that every gunicorn worker resolves the same path for a given upload.
    import hashlib
    original = secure_filename(resumable_filename)
    root, ext = os.path.splitext(original)
    tag = hashlib.md5(f"{resumable_filename}:{total_chunks}".encode()).hexdigest()[:8]
    safe_name = f"{root}_{tag}{ext.lower()}"
    full_file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    # On first chunk, start fresh in case of re-upload
    if resumable_chunk_number == 1 and os.path.exists(full_file_path):
        os.remove(full_file_path)

    try:
        with open(full_file_path, 'ab') as f:
            f.write(chunk.read())
    except Exception as e:
        print(f"Chunk write error: {e}")
        return jsonify({"status": "Error writing chunk to file"}), 400

    print(f"Chunk {resumable_chunk_number}/{total_chunks} for {safe_name}")

    if resumable_chunk_number == total_chunks:
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
    

# Catch-all: serve React index.html for client-side routing (production only)
if _static_dir:
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_react(path):
        if path and os.path.exists(os.path.join(app.static_folder, path)):
            return app.send_static_file(path)
        return app.send_static_file('index.html')


if __name__ == "__main__":
    app.run(host=os.environ.get('FLASK_HOST', '127.0.0.1'), debug=True)

