import { useState } from "react";

const ImageSubmission = () => {
    const [image, setSelectedImage] = useState(null);
    const [uploadedFilename, setUploadedFilename] = useState("");
    const [previewResult, setPreviewResult] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");

    const displayedResult = analysisResult || previewResult;

    const handleDownloadCsv = async () => {
        if (!uploadedFilename) {
            setErrorMessage("No uploaded file available for CSV export.");
            return;
        }

        try {
            setErrorMessage("");
            const response = await fetch(`http://127.0.0.1:5000/download-single/${encodeURIComponent(uploadedFilename)}`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`CSV download failed: ${errorText}`);
            }

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

    const handleRunPipeline = async (event) => {
        event.preventDefault();
        console.log("SUBMITTED");
        setErrorMessage("");
        setAnalysisResult(null);
        setPreviewResult(null);
        setUploadedFilename("");
        
        if (!image) {
            setErrorMessage("Please select an image to upload.");
            return;
        }

        const formData = new FormData();
        formData.append('file', image);

        try {
            setIsUploading(true);
            // 1. Upload the file
            const uploadResponse = await fetch('http://127.0.0.1:5000/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorText = await uploadResponse.text();
                throw new Error(`Upload failed: ${errorText}`);
            }

            const uploadResult = await uploadResponse.json();
            console.log("Upload Success:", uploadResult);

            // 2. Trigger fast preview
            if (uploadResult.filename) {
                console.log("Requesting analysis for:", uploadResult.filename);
                setUploadedFilename(uploadResult.filename);

                const previewResponse = await fetch(`http://127.0.0.1:5000/preview/${encodeURIComponent(uploadResult.filename)}`);
                if (!previewResponse.ok) {
                    const errorText = await previewResponse.text();
                    throw new Error(`Preview failed: ${errorText}`);
                }

                const previewData = await previewResponse.json();
                setPreviewResult(previewData);

                // 3. Trigger full diagnosis as part of same action
                setIsAnalyzing(true);
                const analyzeResponse = await fetch(`http://127.0.0.1:5000/analyze/${encodeURIComponent(uploadResult.filename)}`);
                if (!analyzeResponse.ok) {
                    const errorText = await analyzeResponse.text();
                    throw new Error(`Analyze failed: ${errorText}`);
                }
                const analyzeData = await analyzeResponse.json();
                setAnalysisResult(analyzeData);
                console.log("Analysis Result:", analyzeData);
            } else {
                throw new Error("Upload succeeded but no filename was returned by backend.");
            }

        } catch (error) {
            console.error('Error:', error);
            setErrorMessage(error.message || 'Upload/preview failed.');
        } finally {
            setIsUploading(false);
            setIsAnalyzing(false);
        }
    };

    return ( 
        <div className="diagnosis-layout">
            <div className="preview-panel">
                <h2>Image Preview</h2>
                <div className="comparison-grid">
                    <div className="preview-card">
                        <h3>Original PSR Staining</h3>
                        {displayedResult?.original_image ? (
                            <img alt="Original PSR" src={displayedResult.original_image} className="preview-image" />
                        ) : image ? (
                            <img alt="Selected" src={URL.createObjectURL(image)} className="preview-image" />
                        ) : (
                            <div className="preview-placeholder">Upload a file to preview</div>
                        )}
                    </div>
                    <div className="preview-card">
                        <h3>Fibrosis</h3>
                        {displayedResult?.filtered_image ? (
                            <img alt="Fibrosis mask" src={displayedResult.filtered_image} className="preview-image" />
                        ) : (
                            <div className="preview-placeholder">Fibrosis mask appears after preview</div>
                        )}
                    </div>
                </div>

                <form className="upload-form" onSubmit={handleRunPipeline}>
                    <input
                        type="file"
                        name="myImage"
                        accept=".svs,.tif,.tiff,.jpg,.jpeg,.png,.bmp"
                        onChange={(event) => {
                            setSelectedImage(event.target.files[0]);
                            setErrorMessage("");
                            setPreviewResult(null);
                            setAnalysisResult(null);
                            setUploadedFilename("");
                        }}
                    />
                    <button type="submit" disabled={isUploading || isAnalyzing || !image}>
                        {isUploading || isAnalyzing ? 'Processing...' : 'Upload, Preview & Diagnose'}
                    </button>
                </form>
            </div>

            <aside className="diagnosis-panel">
                <h2>Diagnosis</h2>
                <div className="diag-card">
                    <p className="diag-label">Fibrosis Extent</p>
                    <p className="diag-value">
                        {displayedResult?.fibrosis_ratio !== undefined
                            ? `${Number(displayedResult.fibrosis_ratio).toFixed(2)}%`
                            : '--'}
                    </p>
                </div>
                <div className="diag-card">
                    <p className="diag-label">Classification</p>
                    <p className="diag-classification">
                        {analysisResult?.cluster_label || (isUploading || isAnalyzing ? 'Running diagnosis...' : 'Will run automatically after upload')}
                    </p>
                </div>

                <button
                    type="button"
                    className="diagnose-btn"
                    onClick={handleDownloadCsv}
                    disabled={!uploadedFilename || isUploading || isAnalyzing}
                >
                    Download CSV
                </button>

                {uploadedFilename && <p className="small-note">Uploaded as: {uploadedFilename}</p>}
                {errorMessage && <p className="error-text">{errorMessage}</p>}
            </aside>
        </div>

    );
}
 
export default ImageSubmission;