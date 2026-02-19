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
    const fileInputRef = useRef(null);

    const displayedResult = analysisResult || previewResult;

    const handleFile = useCallback((file) => {
        if (!file) return;
        setSelectedImage(file);
        setErrorMessage("");
        setPreviewResult(null);
        setAnalysisResult(null);
        setUploadedFilename("");
    }, []);

    const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
    const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        handleFile(file);
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

    const handleRunPipeline = async () => {
        setErrorMessage("");
        setAnalysisResult(null);
        setPreviewResult(null);
        setUploadedFilename("");
        if (!image) { setErrorMessage("Please select an image to upload."); return; }

        const formData = new FormData();
        formData.append('file', image);

        try {
            setIsUploading(true);
            const uploadResponse = await fetch('http://127.0.0.1:5000/upload', { method: 'POST', body: formData });
            if (!uploadResponse.ok) throw new Error(`Upload failed: ${await uploadResponse.text()}`);
            const uploadResult = await uploadResponse.json();
            if (!uploadResult.filename) throw new Error("Upload succeeded but no filename returned.");

            setUploadedFilename(uploadResult.filename);

            const previewResponse = await fetch(`http://127.0.0.1:5000/preview/${encodeURIComponent(uploadResult.filename)}`);
            if (!previewResponse.ok) throw new Error(`Preview failed: ${await previewResponse.text()}`);
            setPreviewResult(await previewResponse.json());

            setIsAnalyzing(true);
            const analyzeResponse = await fetch(`http://127.0.0.1:5000/analyze/${encodeURIComponent(uploadResult.filename)}`);
            if (!analyzeResponse.ok) throw new Error(`Analyze failed: ${await analyzeResponse.text()}`);
            const analyzeData = await analyzeResponse.json();
            setAnalysisResult(analyzeData);
        } catch (error) {
            console.error('Error:', error);
            setErrorMessage(error.message || 'Upload/preview failed.');
        } finally {
            setIsUploading(false);
            setIsAnalyzing(false);
        }
    };

    const fibrosisRatio = displayedResult?.fibrosis_ratio;
    const pipelineStatus = analysisResult
        ? 'Diagnosis Complete'
        : isAnalyzing
        ? 'Running Diagnosis…'
        : isUploading
        ? 'Uploading…'
        : image
        ? 'Ready — Click Diagnose'
        : 'Upload a file to start';

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
                            <img alt="Selected" src={URL.createObjectURL(image)} className="preview-image" />
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

                    <button
                        className="run-btn"
                        onClick={handleRunPipeline}
                        disabled={isUploading || isAnalyzing || !image}
                    >
                        {pipelineStatus}
                    </button>
                </div>

                {errorMessage && <p className="error-text">{errorMessage}</p>}
            </div>

            {/* ── Right: diagnosis report ── */}
            <aside className="report-col">
                <h2 className="report-title">DIAGNOSIS REPORT</h2>

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