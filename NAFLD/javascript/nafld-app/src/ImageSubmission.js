import { useState, useRef, useCallback } from "react";

const ImageSubmission = () => {
    const [image, setSelectedImage] = useState(null);
    const [uploadedFilename, setUploadedFilename] = useState("");
    const [previewResult, setPreviewResult] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const [isDragging, setIsDragging] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [patchProgress, setPatchProgress] = useState(null); // { current, total, tissue_patches }
    const fileInputRef = useRef(null);

    const displayedResult = analysisResult || previewResult;

    const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB per chunk
    const LARGE_FILE_THRESHOLD = 10 * 1024 * 1024; // 10 MB — use chunking above this

    // ── Chunked upload for large files (SVS, big TIF) ──
    const uploadChunked = async (file) => {
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        let serverFilename = null;

        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const blob = file.slice(start, end);

            const form = new FormData();
            form.append('file', blob, file.name);
            form.append('resumableFilename', file.name);
            form.append('resumableChunkNumber', String(i + 1));
            form.append('resumableTotalChunks', String(totalChunks));

            const res = await fetch('http://127.0.0.1:5000/largefile', { method: 'POST', body: form });
            if (!res.ok) throw new Error(`Chunk ${i + 1}/${totalChunks} failed: ${await res.text()}`);

            const data = await res.json();
            setUploadProgress(Math.round(((i + 1) / totalChunks) * 100));

            // Last chunk returns the filename
            if (data.filename) serverFilename = data.filename;
        }

        if (!serverFilename) throw new Error('Chunked upload completed but no filename returned.');
        return serverFilename;
    };

    // ── Simple upload for small files ──
    const uploadSimple = async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('http://127.0.0.1:5000/upload', { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`Upload failed: ${await res.text()}`);
        const data = await res.json();
        if (!data.filename) throw new Error('Upload succeeded but no filename returned.');
        setUploadProgress(100);
        return data.filename;
    };

    const handleFile = useCallback(async (file) => {
        if (!file) return;
        setSelectedImage(file);
        setErrorMessage("");
        setPreviewResult(null);
        setAnalysisResult(null);
        setUploadedFilename("");
        setUploadProgress(0);

        try {
            setIsUploading(true);

            // Choose upload strategy based on file size
            const filename = file.size > LARGE_FILE_THRESHOLD
                ? await uploadChunked(file)
                : await uploadSimple(file);

            setUploadedFilename(filename);

            // Fetch preview immediately
            const previewResponse = await fetch(`http://127.0.0.1:5000/preview/${encodeURIComponent(filename)}`);
            if (previewResponse.ok) {
                setPreviewResult(await previewResponse.json());
            }

            // Trigger analysis — use SSE stream for real-time patch progress
            setIsUploading(false);
            setIsAnalyzing(true);
            setPatchProgress(null);

            const isSvsTif = file.name.match(/\.(svs|tif|tiff)$/i);

            if (isSvsTif) {
                // Stream patch progress via SSE
                const analyzeResult = await new Promise((resolve, reject) => {
                    const evtSource = new EventSource(`http://127.0.0.1:5000/analyze-stream/${encodeURIComponent(filename)}`);
                    evtSource.onmessage = (event) => {
                        try {
                            const msg = JSON.parse(event.data);
                            if (msg.type === 'progress') {
                                setPatchProgress({ current: msg.current, total: msg.total, tissue_patches: msg.tissue_patches });
                            } else if (msg.type === 'result') {
                                evtSource.close();
                                resolve(msg.data);
                            }
                        } catch (e) {
                            evtSource.close();
                            reject(e);
                        }
                    };
                    evtSource.onerror = () => {
                        evtSource.close();
                        reject(new Error('Analysis stream connection lost'));
                    };
                });
                setAnalysisResult(analyzeResult);
            } else {
                const analyzeResponse = await fetch(`http://127.0.0.1:5000/analyze/${encodeURIComponent(filename)}`);
                if (!analyzeResponse.ok) throw new Error(`Analyze failed: ${await analyzeResponse.text()}`);
                setAnalysisResult(await analyzeResponse.json());
            }
        } catch (error) {
            console.error("Pipeline error:", error);
            setErrorMessage(error.message || "An error occurred during processing.");
        } finally {
            setIsUploading(false);
            setIsAnalyzing(false);
            setPatchProgress(null);
        }
    }, []);

    const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
    const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    };
    const handleClick = () => fileInputRef.current?.click();

    const handleDownloadCsv = async () => {
        if (!uploadedFilename) { setErrorMessage("No uploaded file available for CSV export."); return; }
        try {
            setErrorMessage("");
            const response = await fetch(`http://127.0.0.1:5000/download-single/${encodeURIComponent(uploadedFilename)}`);
            if (!response.ok) throw new Error(`CSV download failed: ${await response.text()}`);
            const blob = await response.blob();
            const disposition = response.headers.get('Content-Disposition');
            let downloadFilename = `${uploadedFilename}.csv`;
            if (disposition && disposition.includes('filename=')) {
                downloadFilename = disposition.split('filename=')[1].replace(/"/g, '').trim();
            }
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', downloadFilename);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error:', error);
            setErrorMessage(error.message || 'CSV download failed.');
        }
    };

    const fibrosisRatio = displayedResult?.fibrosis_ratio;
    // Simple statuses for the UI label
    const pipelineStatus = isAnalyzing
        ? 'Running Diagnosis…'
        : isUploading
        ? `Uploading${uploadProgress > 0 && uploadProgress < 100 ? '… ' + uploadProgress + '%' : '…'}`
        : analysisResult
        ? 'Diagnosis Complete'
        : null;

    return (
        <div className="main-grid">
            {/* ── Left: image panels ── */}
            <div className="images-col">
                <div className="comparison-grid">
                    {/* Original */}
                    <div
                        className={`img-panel drop-zone ${isDragging ? 'drag-over' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={handleClick}
                    >
                        <span className="img-label">Original PSR Staining</span>
                        {displayedResult?.original_image ? (
                            <img alt="Original PSR" src={displayedResult.original_image} className="preview-image" />
                        ) : image ? (
                            image.name.match(/\.(tif|tiff|svs)$/i) ? (
                                <div className="placeholder-text" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center' }}>
                                    <span style={{ fontSize: '1.5rem' }}>⌛</span>
                                    <span>{image.name}</span>
                                    <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>Generating preview...</span>
                                </div>
                            ) : (
                                <img 
                                    alt="Selected" 
                                    src={URL.createObjectURL(image)} 
                                    className="preview-image" 
                                />
                            )
                        ) : (
                            <div className="placeholder-text">Click or Drop to Upload</div>
                        )}
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".svs,.tif,.tiff,.jpg,.jpeg,.png,.bmp"
                            style={{ display: 'none' }}
                            onChange={(e) => handleFile(e.target.files[0])}
                        />
                    </div>

                    {/* Fibrosis mask */}
                    <div className="img-panel">
                        <span className="img-label accent">AI Fibrosis Mask</span>
                        {displayedResult?.filtered_image ? (
                            <img alt="Fibrosis mask" src={displayedResult.filtered_image} className="preview-image" />
                        ) : (
                            <div className="placeholder-text">Fibrosis mask appears after analysis</div>
                        )}
                    </div>
                </div>

                {/* Bottom bar */}
                <div className="bottom-bar">
                    <div className="file-info">
                        <span className="file-icon">📄</span>
                        <span className="file-name">{image ? image.name : 'No file selected'}</span>
                        <div
                            className="change-link"
                            onClick={handleClick}
                        >
                            Click or Drop to Change
                        </div>
                    </div>

                    <div className="status-indicator" style={{ color: isAnalyzing ? '#4ecdc4' : '#8a9bae', fontSize: '0.85rem' }}>
                        {(isAnalyzing || isUploading) && <span style={{ marginRight: '0.5rem' }}>⏳</span>}
                        {pipelineStatus}
                    </div>
                </div>

                {errorMessage && <p className="error-text">{errorMessage}</p>}
            </div>

            {/* ── Right: diagnosis report ── */}
            <aside className="report-col">
                <h2 className="report-title">DIAGNOSIS REPORT</h2>

                {/* Patch progress card — only visible during SVS/TIF analysis */}
                {patchProgress && isAnalyzing && (
                    <div className="report-card" style={{ borderColor: '#4ecdc4' }}>
                        <p className="report-label" style={{ color: '#4ecdc4', fontWeight: 600 }}>Patch Analysis in Progress</p>
                        <div className="extent-bar-track" style={{ height: '6px', marginBottom: '0.5rem' }}>
                            <div
                                className="extent-bar-fill"
                                style={{ width: `${Math.round((patchProgress.current / patchProgress.total) * 100)}%` }}
                            />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#c5cdd5' }}>
                            <span>Tile {patchProgress.current} / {patchProgress.total}</span>
                            <span>{patchProgress.tissue_patches} tissue patches found</span>
                        </div>
                        <p style={{ fontSize: '0.72rem', color: '#8a9bae', marginTop: '0.3rem' }}>
                            Processing 512×512 patches, skipping blank areas…
                        </p>
                    </div>
                )}

                <div className="report-card">
                    <p className="report-label">Fibrosis Extent</p>
                    <div className="extent-bar-track">
                        <div
                            className="extent-bar-fill"
                            style={{ width: `${Math.min(fibrosisRatio ?? 0, 100)}%` }}
                        />
                    </div>
                    <p className="report-value">
                        {fibrosisRatio !== undefined ? `${Number(fibrosisRatio).toFixed(2)}%` : '--'}
                    </p>
                </div>

                <div className="report-card">
                    <p className="report-label">Classification</p>
                    <p className="report-class">
                        {analysisResult?.cluster_label
                            ? analysisResult.cluster_label.replace(/^Category \w: /, 'Category $&'.slice(9, 10) + ': ').length
                                ? analysisResult.cluster_label
                                : '--'
                            : '--'}
                    </p>
                </div>

                <div className="report-card info-card">
                    <p>
                        <strong>Visualization:</strong> White pixels indicate detected collagen fibers
                        (fibrosis) via Fuzzy C-Means clustering on the VGG16 feature space.
                    </p>
                    {analysisResult?.patch_count && (
                        <p style={{ marginTop: '0.4rem', fontSize: '0.72rem', opacity: 0.7 }}>
                            Patch-based analysis: {analysisResult.patch_count} tissue patches processed
                        </p>
                    )}
                </div>

                <button
                    className="csv-btn"
                    onClick={handleDownloadCsv}
                    disabled={!uploadedFilename || isUploading || isAnalyzing}
                >
                    Download CSV
                </button>
            </aside>
        </div>
    );
};

export default ImageSubmission;