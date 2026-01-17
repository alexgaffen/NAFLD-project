/* src/ImageSubmission.js */
import { useState, useRef } from "react";

const DEMO_MODE = true; 

const ImageSubmission = () => {
    const [image, setSelectedImage] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    
    const fileInputRef = useRef(null);

    // --- FILE HANDLING ---
    const processFile = (file) => {
        if (!file) return;
        if (file.type.startsWith('image/') || file.name.match(/\.(jpg|jpeg|png|tif|tiff)$/i)) {
            setSelectedImage(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResults(null); 
        } else {
            alert("Please upload a valid image file.");
        }
    };

    const handleImageChange = (event) => {
        if (event.target.files && event.target.files[0]) {
            processFile(event.target.files[0]);
        }
    };

    // --- DRAG AND DROP HANDLERS ---
    const onDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const onDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const onDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!image) return;

        setIsLoading(true);
        if (DEMO_MODE) {
            setTimeout(() => {
                setResults({
                    fibrosis_percentage: (Math.random() * 15 + 5).toFixed(2),
                    cluster_label: "Category C: Bridging Fibrosis"
                });
                setIsLoading(false);
            }, 3000); 
            return; 
        }
        
        const formData = new FormData();
        formData.append('file', image);
        try {
            const response = await fetch('http://127.0.0.1:5000/upload', { method: 'POST', body: formData });
            if (!response.ok) throw new Error("Server failed");
            const result = await response.json();
            setResults(result); 
        } catch (error) {
            console.error(error);
            alert("Backend Error");
            setIsLoading(false);
        }
    };

    return ( 
        <div className="dashboard-container">
            {/* ROW 1: HEADER */}
            <div className="top-bar">
                <div className="brand-corner">
                    <div className="logo-icon">🧬</div>
                    <h2 className="logo-text">AIFIBROSIS</h2>
                    <span className="logo-divider">|</span>
                    <span className="app-subtitle">
                        AI-based unsupervised classification and quantification of mouse liver fibrosis in MASH
                    </span>
                </div>
            </div>

            {/* ROW 2: WORKSPACE */}
            <div className="workspace">
                
                {/* LEFT PANEL */}
                <div className="image-viewer">
                    <div className="scientific-header">
                        Quantification Pipeline: VGG16 Feature Extraction → Fuzzy C-Means Clustering
                    </div>

                    {!image ? (
                        /* STATE A: EMPTY UPLOAD */
                        <div className="upload-placeholder">
                            <label 
                                htmlFor="file-upload" 
                                className={`main-upload-area ${isDragging ? 'dragging-active' : ''}`}
                                onDragOver={onDragOver}
                                onDragLeave={onDragLeave}
                                onDrop={onDrop}
                            >
                                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>
                                    {isDragging ? "⏬" : "🔬"}
                                </div>
                                <h3>{isDragging ? "Drop Patch Here" : "Upload Diagnostic Patch"}</h3>
                                <p style={{color:'#94a3b8', fontSize:'0.9rem'}}>
                                    Click to Browse or Drag & Drop<br/>
                                    (PNG, JPG, TIF)
                                </p>
                            </label>
                        </div>
                    ) : (
                        /* STATE B: ACTIVE VIEW */
                        <div className="active-view">
                            
                            {/* --- THE SPLIT FRAME --- */}
                            <div className="scan-frame-split">
                                
                                {/* LEFT BOX: Image + Scoped Drag & Drop */}
                                <div 
                                    className={`split-pane ${isDragging ? 'dragging-active' : ''}`}
                                    onDragOver={onDragOver}
                                    onDragLeave={onDragLeave}
                                    onDrop={onDrop}
                                    title="Drag & Drop to replace patch"
                                >
                                    <div className="viewer-label">Original PSR Staining</div>
                                    <img src={previewUrl} alt="Original" className="scan-img" />
                                    {isLoading && <div className="scanning-laser-horizontal"></div>}
                                </div>

                                {/* RIGHT BOX: Result */}
                                <div className="split-pane">
                                    {results ? (
                                        <>
                                            <div className="viewer-label label-accent">AI Fibrosis Mask</div>
                                            <img src={previewUrl} alt="Mask" className="scan-img mask-simulation" />
                                        </>
                                    ) : (
                                        <div className="empty-mask-placeholder">
                                            <div style={{fontSize:'2rem', opacity:0.3, marginBottom:'10px'}}>⏳</div>
                                            <p>Waiting for user to call diagnosis</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                            
                            {/* --- CONTROLS --- */}
                            <div className="image-controls">
                                <div className="file-info">
                                    <div className="current-filename">
                                        <span style={{fontSize:'1.2rem'}}>📄</span>
                                        {image.name}
                                    </div>
                                    
                                    <label 
                                        htmlFor="re-upload" 
                                        className={`reupload-dropzone ${isDragging ? 'dragging-active' : ''}`}
                                        onDragOver={onDragOver}
                                        onDragLeave={onDragLeave}
                                        onDrop={onDrop}
                                    >
                                        <div style={{fontWeight: 600}}>Click or Drop to Change</div>
                                    </label>
                                    <input 
                                        type="file" 
                                        id="re-upload" 
                                        accept=".png, .jpg, .jpeg, .tif" 
                                        onChange={handleImageChange} 
                                        style={{display: 'none'}} 
                                    />
                                </div>

                                {!results ? (
                                    <button onClick={handleSubmit} className="diagnose-btn" disabled={isLoading}>
                                        {isLoading ? "Processing..." : "▶ Diagnose"}
                                    </button>
                                ) : (
                                    <div className="status-complete">
                                        Diagnosis Complete
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                    
                    <input 
                        type="file" 
                        id="file-upload" 
                        ref={fileInputRef}
                        accept=".png, .jpg, .jpeg, .tif" 
                        onChange={handleImageChange} 
                        style={{display: 'none'}} 
                    />
                </div>

                {/* RIGHT PANEL: RESULTS */}
                <div className="results-sidebar">
                    <div className="sidebar-content-wrapper">
                        <h3 className="sidebar-header">Diagnosis Report</h3>
                        
                        {isLoading ? (
                            <div className="loading-state">
                                <div className="spinner"></div>
                                <p>Quantifying Fibrosis Patterns...</p>
                            </div>
                        ) : results ? (
                            <div className="results-animate-in">
                                {/* Metric 1 */}
                                <div className="metric-box">
                                    <span className="metric-label">Fibrosis Extent</span>
                                    <div className="metric-bar-bg">
                                        <div className="metric-bar-fill" style={{width: `${results.fibrosis_percentage}%`}}></div>
                                    </div>
                                    <span className="metric-value">{results.fibrosis_percentage}%</span>
                                </div>

                                {/* Metric 2 */}
                                <div className="metric-box highlight">
                                    <span className="metric-label">Classification</span>
                                    <div className="classification-badge">{results.cluster_label}</div>
                                </div>

                                <div className="results-note">
                                    <strong>Visualization:</strong> White pixels indicate detected collagen fibers (fibrosis) via Fuzzy C-Means clustering on the VGG16 feature space.
                                </div>
                            </div>
                        ) : (
                            <div className="empty-state">
                                Results will appear here after analysis.
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* ROW 3: FOOTER LOGOS */}
            <div className="bottom-footer">
                <div className="partner-logo-container">
                    <img src="/Images/McMaster.png" alt="McMaster" className="partner-img" />
                    <img src="/Images/ICELAB.png" alt="ICELAB" className="partner-img" />
                    <img src="/Images/Heersink.png" alt="Heersink" className="partner-img" />
                </div>
            </div>
        </div>
    );
}

export default ImageSubmission;