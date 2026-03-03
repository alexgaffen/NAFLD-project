import { useState, useRef, useCallback, useEffect } from "react";

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
    const [tileGrid, setTileGrid] = useState(null); // { rows, cols, tiles: [0=pending|1=bg|2=tissue] }
    const fileInputRef = useRef(null);
    const originalImgRef = useRef(null);
    const [imgContentRect, setImgContentRect] = useState(null); // { left, top, width, height }

    const displayedResult = analysisResult || previewResult;

    // Compute where the object-fit:contain image actually renders inside the panel
    const computeImgRect = useCallback(() => {
        const img = originalImgRef.current;
        if (!img || !img.naturalWidth) return;
        const container = img.parentElement;
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const nw = img.naturalWidth;
        const nh = img.naturalHeight;
        const scale = Math.min(cw / nw, ch / nh);
        const rw = nw * scale;
        const rh = nh * scale;
        setImgContentRect({
            left: (cw - rw) / 2,
            top: (ch - rh) / 2,
            width: rw,
            height: rh,
        });
    }, []);

    // Recompute on resize
    useEffect(() => {
        const img = originalImgRef.current;
        if (!img) return;
        const ro = new ResizeObserver(computeImgRect);
        ro.observe(img.parentElement);
        return () => ro.disconnect();
    }, [computeImgRect, displayedResult]);

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

    const MAX_FLAT_IMAGE_SIZE = 50 * 1024 * 1024; // 50 MB for JPG/PNG/BMP

    const handleFile = useCallback(async (file) => {
        if (!file) return;

        // Enforce 50MB limit on standard flat images — large slides must use SVS or TIF
        const isFlatImage = file.name.match(/\.(jpe?g|png|bmp)$/i);
        if (isFlatImage && file.size > MAX_FLAT_IMAGE_SIZE) {
            setErrorMessage(`Standard images (JPG/PNG/BMP) must be under 50 MB. For whole-slide images, use SVS or TIF format.`);
            return;
        }

        setSelectedImage(file);
        setErrorMessage("");
        setPreviewResult(null);
        setAnalysisResult(null);
        setUploadedFilename("");
        setUploadProgress(0);
        setTileGrid(null);

        try {
            setIsUploading(true);

            // Choose upload strategy based on file size
            const filename = file.size > LARGE_FILE_THRESHOLD
                ? await uploadChunked(file)
                : await uploadSimple(file);

            setUploadedFilename(filename);

            const isSvsTif = file.name.match(/\.(svs|tif|tiff)$/i);

            if (isSvsTif) {
                // Large slides: fetch a quick preview so the user sees something while patches process
                const previewResponse = await fetch(`http://127.0.0.1:5000/preview/${encodeURIComponent(filename)}`);
                if (previewResponse.ok) {
                    setPreviewResult(await previewResponse.json());
                }

                // Stream patch progress via SSE
                setIsUploading(false);
                setIsAnalyzing(true);
                setPatchProgress(null);

                const analyzeResult = await new Promise((resolve, reject) => {
                    const evtSource = new EventSource(`http://127.0.0.1:5000/analyze-stream/${encodeURIComponent(filename)}`);
                    evtSource.onmessage = (event) => {
                        try {
                            const msg = JSON.parse(event.data);
                            if (msg.type === 'progress') {
                                // If the backend sent an analysis-level preview, swap to it
                                // so the tile overlay aligns with the actual image being tiled.
                                if (msg.analysis_preview) {
                                    setPreviewResult(prev => ({
                                        ...prev,
                                        original_image: msg.analysis_preview,
                                    }));
                                }
                                setPatchProgress({ current: msg.current, total: msg.total, tissue_patches: msg.tissue_patches });
                                if (msg.grid_rows !== undefined) {
                                    setTileGrid(prev => {
                                        const r = msg.grid_rows, c = msg.grid_cols;
                                        const tiles = prev && prev.tiles.length === r * c ? [...prev.tiles] : new Array(r * c).fill(0);
                                        tiles[msg.tile_row * c + msg.tile_col] = msg.is_tissue ? 2 : 1;
                                        return { rows: r, cols: c, tiles };
                                    });
                                }
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
                // Small images (JPG/PNG/BMP): skip preview, single analysis call returns everything
                setIsUploading(false);
                setIsAnalyzing(true);
                setPatchProgress(null);

                const analyzeResponse = await fetch(`http://127.0.0.1:5000/analyze/${encodeURIComponent(filename)}`);
                if (!analyzeResponse.ok) throw new Error(`Analyze failed: ${await analyzeResponse.text()}`);
                const result = await analyzeResponse.json();
                setPreviewResult(result);    // Images appear via displayedResult
                setAnalysisResult(result);   // Extent + classification ready simultaneously
            }
        } catch (error) {
            console.error("Pipeline error:", error);
            setErrorMessage(error.message || "An error occurred during processing.");
        } finally {
            setIsUploading(false);
            setIsAnalyzing(false);
            setPatchProgress(null);
            setTileGrid(null);
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

    const handleDownloadMask = () => {
        if (!displayedResult?.filtered_image) return;
        const link = document.createElement('a');
        link.href = displayedResult.filtered_image;
        const baseName = image?.name ? image.name.replace(/\.[^.]+$/, '') : 'fibrosis';
        link.setAttribute('download', `${baseName}_mask.jpg`);
        document.body.appendChild(link);
        link.click();
        link.remove();
    };

    // Only show extent after full analysis — preview images still appear immediately
    const fibrosisRatio = analysisResult?.fibrosis_ratio;
    // Simple statuses for the UI label
    const pipelineStatus = isAnalyzing
        ? 'Running Diagnosis…'
        : isUploading
        ? `Uploading${uploadProgress > 0 && uploadProgress < 100 ? '… ' + uploadProgress + '%' : '…'}`
        : null;

    const renderRadarChart = (data) => {
        if (!data) return null;
        const cx = 150, cy = 150, R = 80;
        const N = 5;
        const cats = [
            { key: 'None',            label: 'None',            sub: 'F0' },
            { key: 'Perisinusoidal',  label: 'Perisinusoidal',  sub: 'F1' },
            { key: 'Periportal',      label: 'Periportal',      sub: 'F2' },
            { key: 'Bridging',        label: 'Bridging',        sub: 'F3' },
            { key: 'Cirrosis',        label: 'Cirrhosis',       sub: 'F4' },
        ];
        const angleOf = (i) => -Math.PI / 2 + (2 * Math.PI * i) / N;
        const pts = cats.map((c, i) => {
            const a = angleOf(i);
            const v = data[c.key] || 0;
            return {
                x: cx + R * v * Math.cos(a),
                y: cy + R * v * Math.sin(a),
                ex: cx + R * Math.cos(a),
                ey: cy + R * Math.sin(a),
                v, label: c.label, sub: c.sub, angle: a, idx: i,
            };
        });
        const poly = pts.map(p => `${p.x},${p.y}`).join(' ');
        // web lines (pentagons at each ring)
        const rings = [0.25, 0.5, 0.75, 1.0];
        const webLines = rings.map(f => {
            const webPts = cats.map((_, i) => {
                const a = angleOf(i);
                return `${cx + R * f * Math.cos(a)},${cy + R * f * Math.sin(a)}`;
            }).join(' ');
            return webPts;
        });
        return (
            <svg viewBox="0 0 300 310" style={{ width: '100%', maxWidth: 280, display: 'block', margin: '0.5rem auto' }}>
                {webLines.map((w, i) => (
                    <polygon key={`web${i}`} points={w}
                        fill="none" stroke="#253545" strokeWidth={0.75}
                        strokeDasharray={rings[i] < 1 ? '4 4' : 'none'} />
                ))}
                {pts.map((p, i) => (
                    <line key={`ax${i}`} x1={cx} y1={cy} x2={p.ex} y2={p.ey}
                        stroke="#4ecdc4" strokeWidth={1.5} opacity={0.45} />
                ))}
                <polygon points={poly} fill="rgba(78,205,196,0.22)" stroke="#4ecdc4" strokeWidth={2} />
                {pts.map((p, i) => (
                    <circle key={`dot${i}`} cx={p.x} cy={p.y} r={4.5}
                        fill="#0f1923" stroke="#4ecdc4" strokeWidth={2} />
                ))}
                {pts.map((p) => {
                    const a = p.angle;
                    const cos = Math.cos(a), sin = Math.sin(a);
                    const isTop = sin < -0.3, isBot = sin > 0.3, isLeft = cos < -0.3, isRight = cos > 0.3;
                    const anchor = isRight ? 'start' : isLeft ? 'end' : 'middle';
                    const dx = isRight ? 14 : isLeft ? -14 : 0;
                    const dy1 = isTop ? -22 : isBot ? 18 : -6;
                    const dy2 = dy1 + 12;
                    return (
                        <g key={`lbl${p.idx}`}>
                            <text x={p.ex + dx} y={p.ey + dy1} fill="#fff" fontSize="15" fontWeight="700"
                                textAnchor={anchor}>{`${(p.v * 100).toFixed(0)}%`}</text>
                            <text x={p.ex + dx} y={p.ey + dy2} fill="#8a9bae" fontSize="12"
                                textAnchor={anchor}>{p.label}</text>
                        </g>
                    );
                })}
            </svg>
        );
    };

    // Should we show the tile overlay on the original image?
    const showTileOverlay = tileGrid && isAnalyzing;

    return (
        <div className="main-grid">
            {/* ── Left: images + extent description ── */}
            <div className="images-col">
                <div className="comparison-grid">
                    {/* Original — with tile overlay during analysis */}
                    <div
                        className={`img-panel drop-zone ${isDragging ? 'drag-over' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={handleClick}
                    >
                        <span className="img-label">Original PSR Staining</span>
                        {displayedResult?.original_image ? (
                            <img ref={originalImgRef} onLoad={computeImgRect} alt="Original PSR" src={displayedResult.original_image} className="preview-image" draggable="false" onDragStart={(e) => e.preventDefault()} />
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
                                    draggable="false"
                                    onDragStart={(e) => e.preventDefault()}
                                />
                            )
                        ) : (
                            <div className="placeholder-text">
                                <div style={{ color: '#ffffff' }}>Click or Drop to Upload</div>
                                <div style={{ fontSize: '0.7rem', opacity: 0.7, marginTop: '0.4rem', color: '#ffffff' }}>SVS / TIF — any size &nbsp;·&nbsp; JPG / PNG / BMP — max 50 MB</div>
                            </div>
                        )}

                        {/* Filename & change message at bottom of frame */}
                        {image && (
                            <div className="img-panel-footer">
                                <span className="img-panel-filename">📄 {image.name}</span>
                                <span className="img-panel-change" onClick={handleClick}>Click or drop to change</span>
                            </div>
                        )}

                        {/* Tile grid overlay — positioned over the actual image content only */}
                        {showTileOverlay && imgContentRect && (
                            <div className="tile-overlay" style={{
                                left: imgContentRect.left,
                                top: imgContentRect.top,
                                width: imgContentRect.width,
                                height: imgContentRect.height,
                                gridTemplateColumns: `repeat(${tileGrid.cols}, 1fr)`,
                                gridTemplateRows: `repeat(${tileGrid.rows}, 1fr)`,
                            }}>
                                {tileGrid.tiles.map((s, i) => (
                                    <div key={i} className={`tile-cell ${s === 2 ? 'tile-tissue' : s === 1 ? 'tile-bg' : 'tile-pending'}`} />
                                ))}
                            </div>
                        )}

                        {/* Green progress bar at bottom of upload image */}
                        {(isUploading || isAnalyzing) && (
                            <div className="img-progress-bar-track">
                                <div
                                    className="img-progress-bar-fill"
                                    style={{
                                        width: isAnalyzing && patchProgress
                                            ? `${Math.round((patchProgress.current / patchProgress.total) * 100)}%`
                                            : isUploading
                                            ? `${uploadProgress}%`
                                            : '0%',
                                    }}
                                />
                            </div>
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
                            <img alt="Fibrosis mask" src={displayedResult.filtered_image} className="preview-image" draggable="false" onDragStart={(e) => e.preventDefault()} />
                        ) : (
                            <div className="placeholder-text">Fibrosis mask appears after analysis</div>
                        )}
                        {displayedResult?.filtered_image && (
                            <button className="download-mask-btn" onClick={handleDownloadMask} title="Download fibrosis mask">
                                ⬇ Save Mask
                            </button>
                        )}
                    </div>
                </div>

                {/* Extent description spanning full width under both images */}
                <div className="extent-description">
                    <div className="report-card info-card">
                        <p>
                            <strong>Visualization & Extent:</strong> White pixels indicate detected collagen fibers
                            isolated via colour deconvolution and adaptive thresholding.
                        </p>
                        <p style={{ marginTop: '0.4rem' }}>
                            <strong>Staging:</strong> Disease category is classified by analyzing the
                            architectural patterns of these fibers using a VGG16 neural network and Fuzzy C-Means clustering.
                        </p>
                        {analysisResult?.patch_count && (
                            <p style={{ marginTop: '0.4rem', fontSize: '0.72rem' }}>
                                Patch-based analysis: {analysisResult.patch_count} tissue patches processed
                            </p>
                        )}
                    </div>
                </div>

                {errorMessage && <p className="error-text">{errorMessage}</p>}
            </div>

            {/* ── Right: diagnosis report ── */}
            <aside className="report-col">
                <div className="report-header">
                    <h2 className="report-title">DIAGNOSIS REPORT</h2>
                    <button
                        className="csv-btn-inline"
                        onClick={handleDownloadCsv}
                        disabled={!uploadedFilename || isUploading || isAnalyzing}
                    >
                        ⬇ Download CSV
                    </button>
                </div>

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

                <div className="report-card" style={{ border: '1px solid #4ecdc4' }}>
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

                {/* FCM Membership Radar Chart — 5 categories + spectrum note */}
                {analysisResult && (analysisResult.None !== undefined) && (
                    <div className="report-card" style={{ border: '1px solid #4ecdc4' }}>
                        <p className="report-label">FCM Cluster Membership</p>
                        {renderRadarChart(analysisResult)}
                        <p style={{ fontSize: '0.78rem', color: '#8a9bae', lineHeight: 1.5, marginTop: '0.25rem', borderTop: '1px solid #253545', paddingTop: '0.4rem' }}>
                            <strong style={{ color: '#f7b731' }}>Gradual Spectrum:</strong>{' '}
                            Fibrosis stages (F0–F4) overlap continuously. FCM assigns probabilistic membership rather than a hard category.
                        </p>
                    </div>
                )}

            </aside>
        </div>
    );
};

export default ImageSubmission;