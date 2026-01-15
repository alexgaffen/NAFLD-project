/* src/ImageSubmission.js */
import { useState } from "react";

const DEMO_MODE = true; 

const ImageSubmission = () => {
    const [image, setSelectedImage] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState(null);

    const handleImageChange = (event) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            setSelectedImage(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResults(null); 
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
            }, 2500);
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
            {/* 1. HEADER */}
            <div className="top-bar">
                <div className="brand-corner">
                    <div className="logo-icon">🧬</div>
                    <h2 className="logo-text">AiFibrosis</h2>
                    
                    <div className="partner-logos">
                        <img src="/Images/McMaster.png" alt="McMaster" className="partner-img" />
                        <img src="/Images/ICELAB.png" alt="ICELAB" className="partner-img" />
                        <img src="/Images/Heersink.png" alt="Heersink" className="partner-img" />
                    </div>
                </div>
            </div>

            <div className="workspace">
                {/* 2. LEFT PANEL */}
                <div className="image-viewer">
                    
                    {/* --- MOVED: Scientific Description is now here --- */}
                    <div className="scientific-header">
                        AI-based unsupervised classification and quantification of mouse liver fibrosis in MASH
                    </div>

                    {!image ? (
                        /* STATE A: EMPTY UPLOAD */
                        <div className="upload-placeholder">
                            <label htmlFor="file-upload" className="main-upload-area">
                                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>🔬</div>
                                
                                <h3>Upload Diagnostic Patch</h3>
                                
                                <p style={{color:'#94a3b8', fontSize:'0.9rem'}}>
                                    Standard Histology Formats (PNG, JPG, TIF)
                                </p>
                            </label>
                        </div>
                    ) : (
                        /* STATE B: ACTIVE VIEW */
                        <div className="active-view">
                            
                            <div className="viewer-header">
                                Image Preview
                            </div>

                            <div className="scan-frame">
                                <img src={previewUrl} alt="Scan" className="scan-img" />
                                {isLoading && <div className="scanning-laser"></div>}
                            </div>
                            
                            <div className="image-controls">
                                <div className="file-info">
                                    <span className="current-filename">📄 {image.name}</span>
                                    <label htmlFor="re-upload" className="reupload-link">Change Patch</label>
                                    <input type="file" id="re-upload" accept=".png, .jpg, .jpeg, .tif" onChange={handleImageChange} style={{display: 'none'}} />
                                </div>
                                <button onClick={handleSubmit} className="diagnose-btn" disabled={isLoading}>
                                    {isLoading ? "Analyzing..." : "▶ Diagnose"}
                                </button>
                            </div>
                        </div>
                    )}
                    <input type="file" id="file-upload" accept=".png, .jpg, .jpeg, .tif" onChange={handleImageChange} style={{display: 'none'}} />
                </div>

                {/* 3. RIGHT PANEL */}
                <div className="results-sidebar">
                    <h3 className="sidebar-header">Diagnosis</h3>
                    
                    {isLoading ? (
                        <div style={{textAlign:'center', marginTop:'50px', color: '#94a3b8'}}>
                            <p>Quantifying Fibrosis Patterns...</p>
                        </div>
                    ) : results ? (
                        <div className="results-animate-in">
                            <div className="metric-box">
                                <span className="metric-label">Fibrosis Extent</span>
                                <div className="metric-bar-bg"><div className="metric-bar-fill" style={{width: `${results.fibrosis_percentage}%`}}></div></div>
                                <span className="metric-value">{results.fibrosis_percentage}%</span>
                            </div>

                            <div className="metric-box highlight">
                                <span className="metric-label">Classification</span>
                                <div className="classification-badge">{results.cluster_label}</div>
                            </div>
                        </div>
                    ) : (
                        <div style={{color: '#64748b', fontStyle: 'italic'}}>
                            Results will appear here after analysis.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default ImageSubmission;