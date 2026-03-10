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
    const filteredImgRef = useRef(null);
    const [imgContentRect, setImgContentRect] = useState(null); // { left, top, width, height }

    // Threshold slider state
    const [autoThreshold, setAutoThreshold] = useState(null);    // original AI threshold
    const [userThreshold, setUserThreshold] = useState(null);    // current slider value
    const [adjustedRatio, setAdjustedRatio] = useState(null);    // ratio from rethreshold
    const [adjustedMask, setAdjustedMask] = useState(null);      // base64 mask from rethreshold
    const [isRethresholding, setIsRethresholding] = useState(false);
    const rethresholdTimer = useRef(null);

    // Magnifier state
    const [magnifier, setMagnifier] = useState(null); // { x, y } normalised 0-1
    const [magnifierZoom, setMagnifierZoom] = useState(1); // default: no zoom
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 6;
    const ZOOM_STEP = 0.5;
    const [hasLocalEdits, setHasLocalEdits] = useState(false);
    const [showResetConfirm, setShowResetConfirm] = useState(false);
    const pendingResetAction = useRef(null);
    const areaDeltaTimer = useRef(null);
    const pendingAreaDelta = useRef(0);
    const [deltaMap, setDeltaMap] = useState(null); // base64 PNG of modified areas
    const [showHelp, setShowHelp] = useState(false);

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

    // When analysis completes, pick up the auto threshold
    useEffect(() => {
        if (analysisResult?.threshold !== undefined) {
            setAutoThreshold(analysisResult.threshold);
            setUserThreshold(analysisResult.threshold);
            setAdjustedRatio(null);
            setAdjustedMask(null);
            setHasLocalEdits(false);
            setDeltaMap(null);
        }
    }, [analysisResult]);

    // Debounced rethreshold call (global — resets local edits)
    const handleThresholdChange = useCallback((value) => {
        setUserThreshold(value);
        if (rethresholdTimer.current) clearTimeout(rethresholdTimer.current);
        rethresholdTimer.current = setTimeout(async () => {
            if (!uploadedFilename) return;
            setIsRethresholding(true);
            try {
                const res = await fetch(
                    `http://127.0.0.1:5000/rethreshold/${encodeURIComponent(uploadedFilename)}?threshold=${value}`
                );
                if (res.ok) {
                    const data = await res.json();
                    setAdjustedRatio(data.fibrosis_ratio);
                    setAdjustedMask(data.filtered_image);
                    setHasLocalEdits(false);
                    setDeltaMap(null);
                }
            } catch (e) {
                console.error('Rethreshold error:', e);
            } finally {
                setIsRethresholding(false);
            }
        }, 150);
    }, [uploadedFilename]);

    // Magnifier: compute normalised position from mouse event on an img-panel
    // Only activates when cursor is over the actual rendered image content
    const handlePanelMouseMove = useCallback((e) => {
        const panel = e.currentTarget;
        const rect = panel.getBoundingClientRect();
        const img = panel.querySelector('img.preview-image');
        if (img && img.naturalWidth) {
            const cw = rect.width;
            const ch = rect.height;
            const nw = img.naturalWidth;
            const nh = img.naturalHeight;
            const scale = Math.min(cw / nw, ch / nh);
            const rw = nw * scale;
            const rh = nh * scale;
            const offsetX = (cw - rw) / 2;
            const offsetY = (ch - rh) / 2;
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            if (mouseX < offsetX || mouseX > offsetX + rw || mouseY < offsetY || mouseY > offsetY + rh) {
                setMagnifier(null);
                return;
            }
        }
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        setMagnifier({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) });
    }, []);
    const handlePanelMouseLeave = useCallback(() => setMagnifier(null), []);

    // Compute the normalised region visible in the magnifier
    const MAGNIFIER_SIZE = 160;
    const getMagnifierRegion = useCallback(() => {
        if (!magnifier) return null;
        const img = originalImgRef.current;
        if (!img) return null;
        const container = img.parentElement;
        if (!container) return null;
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const nw = img.naturalWidth || cw;
        const nh = img.naturalHeight || ch;
        const scale = Math.min(cw / nw, ch / nh);
        const rw = nw * scale;
        const rh = nh * scale;
        const posX = magnifier.x * cw;
        const posY = magnifier.y * ch;
        const offsetX = (cw - rw) / 2;
        const offsetY = (ch - rh) / 2;
        const imgX = (posX - offsetX) / rw;
        const imgY = (posY - offsetY) / rh;
        if (imgX < 0 || imgX > 1 || imgY < 0 || imgY > 1) return null;
        const halfW = MAGNIFIER_SIZE / (2 * rw * magnifierZoom);
        const halfH = MAGNIFIER_SIZE / (2 * rh * magnifierZoom);
        return {
            x1: Math.max(0, imgX - halfW),
            y1: Math.max(0, imgY - halfH),
            x2: Math.min(1, imgX + halfW),
            y2: Math.min(1, imgY + halfH),
        };
    }, [magnifier, magnifierZoom]);

    // Debounced local area delta rethreshold (per-pixel)
    const handleLocalDeltaChange = useCallback((region) => {
        if (areaDeltaTimer.current) clearTimeout(areaDeltaTimer.current);
        areaDeltaTimer.current = setTimeout(async () => {
            if (!uploadedFilename || !region) return;
            const delta = pendingAreaDelta.current;
            pendingAreaDelta.current = 0;
            if (delta === 0) return;
            setIsRethresholding(true);
            try {
                const params = new URLSearchParams({
                    delta,
                    x1: region.x1, y1: region.y1,
                    x2: region.x2, y2: region.y2,
                });
                const res = await fetch(
                    `http://127.0.0.1:5000/rethreshold-area/${encodeURIComponent(uploadedFilename)}?${params}`
                );
                if (res.ok) {
                    const data = await res.json();
                    setAdjustedRatio(data.fibrosis_ratio);
                    setAdjustedMask(data.filtered_image);
                    setHasLocalEdits(true);
                    if (data.delta_map) setDeltaMap(data.delta_map);
                }
            } catch (e) {
                console.error('Rethreshold area error:', e);
            } finally {
                setIsRethresholding(false);
            }
        }, 150);
    }, [uploadedFilename]);

    // Confirmation overlay helpers
    const requestConfirmReset = useCallback((action) => {
        pendingResetAction.current = action;
        setShowResetConfirm(true);
    }, []);
    const handleConfirmYes = useCallback(() => {
        setShowResetConfirm(false);
        if (pendingResetAction.current) {
            pendingResetAction.current();
            pendingResetAction.current = null;
        }
    }, []);
    const handleConfirmNo = useCallback(() => {
        setShowResetConfirm(false);
        pendingResetAction.current = null;
    }, []);

    // Reset area under observation to original AI threshold (Ctrl+R)
    const handleResetArea = useCallback(async () => {
        const region = getMagnifierRegion();
        if (!region || !uploadedFilename) return;
        setIsRethresholding(true);
        try {
            const params = new URLSearchParams({
                x1: region.x1, y1: region.y1,
                x2: region.x2, y2: region.y2,
            });
            const res = await fetch(
                `http://127.0.0.1:5000/reset-area/${encodeURIComponent(uploadedFilename)}?${params}`
            );
            if (res.ok) {
                const data = await res.json();
                setAdjustedRatio(data.fibrosis_ratio);
                setAdjustedMask(data.filtered_image);
                setHasLocalEdits(data.has_local_edits);
                setDeltaMap(data.delta_map);
            }
        } catch (e) {
            console.error('Reset area error:', e);
        } finally {
            setIsRethresholding(false);
        }
    }, [uploadedFilename, getMagnifierRegion]);

    // Undo last modified square (Ctrl+Z)
    const handleUndoArea = useCallback(async () => {
        if (!uploadedFilename) return;
        setIsRethresholding(true);
        try {
            const res = await fetch(
                `http://127.0.0.1:5000/undo-area/${encodeURIComponent(uploadedFilename)}`
            );
            if (res.ok) {
                const data = await res.json();
                setAdjustedRatio(data.fibrosis_ratio);
                setAdjustedMask(data.filtered_image);
                setHasLocalEdits(data.has_local_edits);
                setDeltaMap(data.delta_map);
            }
        } catch (e) {
            console.error('Undo area error:', e);
        } finally {
            setIsRethresholding(false);
        }
    }, [uploadedFilename]);

    // Key handler: Ctrl+Z = undo (anytime), Ctrl+R = reset area (magnifier active),
    // arrows = zoom & area delta (magnifier active), Escape = close overlays
    useEffect(() => {
        const handler = (e) => {
            // Escape closes help or confirm overlays
            if (e.key === 'Escape') {
                if (showHelp) { setShowHelp(false); return; }
                if (showResetConfirm) { handleConfirmNo(); return; }
            }
            // Ctrl+Z: undo last modified square (works anytime)
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                handleUndoArea();
                return;
            }
            // Everything below requires active magnifier
            if (!magnifier) return;
            // Ctrl+R: reset area under observation
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                handleResetArea();
                return;
            }
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                e.stopPropagation();
                setMagnifierZoom(z => Math.min(z + ZOOM_STEP, MAX_ZOOM));
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                e.stopPropagation();
                setMagnifierZoom(z => Math.max(z - ZOOM_STEP, MIN_ZOOM));
            } else if (e.key === 'ArrowLeft' && autoThreshold !== null) {
                e.preventDefault();
                e.stopPropagation();
                const region = getMagnifierRegion();
                if (region) {
                    pendingAreaDelta.current -= 0.05;
                    handleLocalDeltaChange(region);
                }
            } else if (e.key === 'ArrowRight' && autoThreshold !== null) {
                e.preventDefault();
                e.stopPropagation();
                const region = getMagnifierRegion();
                if (region) {
                    pendingAreaDelta.current += 0.05;
                    handleLocalDeltaChange(region);
                }
            }
        };
        window.addEventListener('keydown', handler, true);
        return () => window.removeEventListener('keydown', handler, true);
    }, [magnifier, autoThreshold, getMagnifierRegion, handleLocalDeltaChange, handleResetArea, handleUndoArea, showHelp, showResetConfirm, handleConfirmNo]);

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
        setAutoThreshold(null);
        setUserThreshold(null);
        setAdjustedRatio(null);
        setAdjustedMask(null);
        setMagnifier(null);
        setHasLocalEdits(false);
        setShowResetConfirm(false);
        setDeltaMap(null);

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

    const isBusy = isUploading || isAnalyzing;
    const handleDragOver = (e) => { e.preventDefault(); if (!isBusy) setIsDragging(true); };
    const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (isBusy) return;
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    };
    const handleClick = () => { if (!isBusy) fileInputRef.current?.click(); };

    const handleDownloadCsv = async () => {
        if (!uploadedFilename) { setErrorMessage("No uploaded file available for CSV export."); return; }
        try {
            setErrorMessage("");
            let csvUrl = `http://127.0.0.1:5000/download-single/${encodeURIComponent(uploadedFilename)}`;
            if (adjustedRatio !== null) {
                csvUrl += `?fibrosis_ratio=${adjustedRatio}`;
            }
            const response = await fetch(csvUrl);
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
        const maskSrc = adjustedMask || displayedResult?.filtered_image;
        if (!maskSrc) return;
        const link = document.createElement('a');
        link.href = maskSrc;
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

    // The mask to display: use adjusted mask if user moved slider, else the original
    const displayedMaskSrc = adjustedMask || displayedResult?.filtered_image;

    // Should we show the tile overlay on the original image?
    const showTileOverlay = tileGrid && isAnalyzing;

    // Magnifier rendering helper
    const zoomFraction = (magnifierZoom - MIN_ZOOM) / (MAX_ZOOM - MIN_ZOOM);
    const renderMagnifier = (imgRef) => {
        if (!magnifier || !imgRef.current) return null;
        const img = imgRef.current;
        const container = img.parentElement;
        if (!container) return null;
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const posX = magnifier.x * cw;
        const posY = magnifier.y * ch;
        const nw = img.naturalWidth || cw;
        const nh = img.naturalHeight || ch;
        const scale = Math.min(cw / nw, ch / nh);
        const rw = nw * scale;
        const rh = nh * scale;
        const offsetX = (cw - rw) / 2;
        const offsetY = (ch - rh) / 2;
        const imgX = (posX - offsetX) / rw;
        const imgY = (posY - offsetY) / rh;
        if (imgX < 0 || imgX > 1 || imgY < 0 || imgY > 1) return null;
        const effectiveZoom = Math.max(magnifierZoom, 1.001);
        const bgW = nw * effectiveZoom * scale;
        const bgH = nh * effectiveZoom * scale;
        const bgX = -(imgX * bgW - MAGNIFIER_SIZE / 2);
        const bgY = -(imgY * bgH - MAGNIFIER_SIZE / 2);
        return (
            <div className="magnifier-group" style={{
                left: posX - MAGNIFIER_SIZE / 2,
                top: posY - MAGNIFIER_SIZE / 2,
            }}>
                <div className="mag-main-col">
                    <div className="mag-top-row">
                        <div className="magnifier-lens" style={{
                            width: MAGNIFIER_SIZE,
                            height: MAGNIFIER_SIZE,
                            backgroundImage: `url(${img.src})`,
                            backgroundSize: `${bgW}px ${bgH}px`,
                            backgroundPosition: `${bgX}px ${bgY}px`,
                        }} />
                        {/* Right sidebar: zoom */}
                        <div className="magnifier-sidebar">
                            <span className="mag-sidebar-label">Zoom</span>
                            <div className="mag-zoom-bar">
                                <span className="mag-arrow-hint">▲</span>
                                <div className="mag-zoom-track">
                                    <div className="mag-zoom-fill" style={{ height: `${zoomFraction * 100}%` }} />
                                </div>
                                <span className="mag-arrow-hint">▼</span>
                            </div>
                        </div>
                    </div>
                    {/* Bottom bar: area adjustment */}
                    <div className="mag-bottom-bar">
                        <span className="mag-arrow-hint">◀</span>
                        <div className="mag-adj-track">
                            <div className="mag-adj-center" />
                        </div>
                        <span className="mag-arrow-hint">▶</span>
                        <span className="mag-bottom-label">Area Adj.</span>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="main-grid">
            {/* Help button — fixed top-right */}
            <button className="help-btn" onClick={() => setShowHelp(true)} title="Help &amp; Keybindings">?</button>

            {/* ── Left: images + extent description ── */}
            <div className="images-col">
                <div className="comparison-grid">
                    {/* Original — with tile overlay during analysis */}
                    <div
                        className={`img-panel drop-zone ${isDragging ? 'drag-over' : ''} ${magnifier && displayedResult?.original_image ? 'magnifier-active' : ''} ${isBusy ? 'busy' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={handleClick}
                        onMouseMove={displayedResult?.original_image ? handlePanelMouseMove : undefined}
                        onMouseLeave={displayedResult?.original_image ? handlePanelMouseLeave : undefined}
                    >
                        <span className="img-label">Original PSR Staining</span>
                        {displayedResult?.original_image ? (
                            <img ref={originalImgRef} onLoad={computeImgRect} alt="Original PSR" src={displayedResult.original_image} className="preview-image" draggable="false" onDragStart={(e) => e.preventDefault()} />
                        ) : image ? (
                            image.name.match(/\.(tif|tiff|svs)$/i) ? (
                                <div className="placeholder-text" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center' }}>
                                    <span style={{ fontSize: '1.5rem' }}>⌛</span>
                                    <span>{image.name}</span>
                                    <span style={{ fontSize: '0.75rem' }}>Generating preview...</span>
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
                                <div style={{ fontSize: '0.7rem', marginTop: '0.4rem', color: '#ffffff' }}>SVS / TIF — any size &nbsp;·&nbsp; JPG / PNG / BMP — max 50 MB</div>
                            </div>
                        )}

                {/* Magnifier overlay */}
                        {magnifier && displayedResult?.original_image && renderMagnifier(originalImgRef)}
                        {magnifier && displayedResult?.original_image && <div className="magnifier-dim" />}

                        {/* Filename & change message at bottom of frame */}
                        {image && (
                            <div className="img-panel-footer">
                                <span className="img-panel-filename">📄 {image.name}</span>
                                {!isBusy && <span className="img-panel-change" onClick={handleClick}>Click or drop to change</span>}
                                {isAnalyzing && <button className="cancel-diagnosis-btn" onClick={() => window.location.reload()}>✕ Cancel Diagnosis</button>}
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
                            disabled={isBusy}
                            onChange={(e) => handleFile(e.target.files[0])}
                        />
                    </div>

                    {/* Fibrosis mask */}
                    <div
                        className={`img-panel ${magnifier && displayedMaskSrc ? 'magnifier-active' : ''}`}
                        onMouseMove={displayedMaskSrc ? handlePanelMouseMove : undefined}
                        onMouseLeave={displayedMaskSrc ? handlePanelMouseLeave : undefined}
                    >
                        <span className="img-label accent">AI Fibrosis Mask</span>
                        {displayedMaskSrc ? (
                            <img ref={filteredImgRef} alt="Fibrosis mask" src={displayedMaskSrc} className="preview-image" draggable="false" onDragStart={(e) => e.preventDefault()} />
                        ) : (
                            <div className="placeholder-text">Fibrosis mask appears after analysis</div>
                        )}
                        {magnifier && displayedMaskSrc && renderMagnifier(filteredImgRef)}
                        {magnifier && displayedMaskSrc && <div className="magnifier-dim" />}
                        {displayedMaskSrc && (
                            <button className="download-mask-btn" onClick={handleDownloadMask} title="Download fibrosis mask">
                                ⬇ Save Mask
                            </button>
                        )}
                    </div>
                </div>

                {/* Area-only adjustment message + binding indicators (replaces extent description when magnifier active) */}
                <div className="extent-description">
                    {magnifier && displayedResult?.original_image ? (
                        <div className="binding-indicators">
                            <p className="magnifier-area-msg">◀ ▶ magnifying glass threshold adjustments apply to the area under observation only</p>
                            <div className="binding-row">
                                <div className="binding-group">
                                    <span className="binding-key binding-key-yellow">▲</span>
                                    <span className="binding-label">Zoom In</span>
                                </div>
                                <div className="binding-group">
                                    <span className="binding-key binding-key-yellow">▼</span>
                                    <span className="binding-label">Zoom Out</span>
                                </div>
                                <div className="binding-group">
                                    <span className="binding-key">◀</span>
                                    <span className="binding-label">− Threshold</span>
                                </div>
                                <div className="binding-group">
                                    <span className="binding-key">▶</span>
                                    <span className="binding-label">+ Threshold</span>
                                </div>
                                <div className="binding-group">
                                    <span className="binding-key binding-key-sm">Ctrl+Z</span>
                                    <span className="binding-label">Undo</span>
                                </div>
                                <div className="binding-group">
                                    <span className="binding-key binding-key-sm">Ctrl+R</span>
                                    <span className="binding-label">Reset Area</span>
                                </div>
                            </div>
                        </div>
                    ) : (
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
                    )}
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
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#fff' }}>
                            <span>Tile {patchProgress.current} / {patchProgress.total}</span>
                            <span>{patchProgress.tissue_patches} tissue patches found</span>
                        </div>
                        <p style={{ fontSize: '0.72rem', color: '#fff', marginTop: '0.3rem' }}>
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
                        {adjustedRatio !== null && (
                            <span className="adjusted-ratio"> → {Number(adjustedRatio).toFixed(2)}%</span>
                        )}
                    </p>
                    {hasLocalEdits && (
                        <div className="local-edits-badge">
                            <span>■ Area adjustments applied</span>
                            <button className="undo-area-btn" onClick={handleUndoArea} title="Undo last area modification (Ctrl+Z)">↩ Undo Step</button>
                        </div>
                    )}
                    {autoThreshold !== null && (
                        <div className="threshold-slider-wrapper">
                            <label className="threshold-label">
                                Baseline Threshold
                                <span className="threshold-value">{Number(userThreshold).toFixed(3)}</span>
                                {isRethresholding && <span className="threshold-loading">⟳</span>}
                            </label>
                            <input
                                type="range"
                                className="threshold-slider"
                                min={Math.max(autoThreshold - 1.5, 0)}
                                max={autoThreshold + 1.5}
                                step={0.01}
                                value={userThreshold}
                                onChange={(e) => {
                                    const v = parseFloat(e.target.value);
                                    if (hasLocalEdits) {
                                        requestConfirmReset(() => handleThresholdChange(v));
                                        return;
                                    }
                                    handleThresholdChange(v);
                                }}
                            />
                            <div className="threshold-hint">
                                <span>More inclusive</span>
                                <span>More restrictive</span>
                            </div>
                            {adjustedRatio !== null && (
                                <button className="threshold-reset-btn" onClick={() => {
                                    const doReset = () => {
                                        setUserThreshold(autoThreshold);
                                        setAdjustedRatio(null);
                                        setAdjustedMask(null);
                                        setHasLocalEdits(false);
                                        setDeltaMap(null);
                                        fetch(`http://127.0.0.1:5000/rethreshold/${encodeURIComponent(uploadedFilename)}?threshold=${autoThreshold}`)
                                            .catch(() => {});
                                    };
                                    if (hasLocalEdits) {
                                        requestConfirmReset(doReset);
                                    } else {
                                        doReset();
                                    }
                                }}>Reset to baseline AI estimate</button>
                            )}
                        </div>
                    )}
                </div>

                {/* FCM Membership Radar Chart — 5 categories + spectrum note */}
                {analysisResult && (analysisResult.None !== undefined) && (
                    <div className="report-card" style={{ border: '1px solid #4ecdc4' }}>
                        <p className="report-label">FCM Cluster Membership</p>
                        {renderRadarChart(analysisResult)}
                        <p style={{ fontSize: '0.78rem', color: '#fff', lineHeight: 1.5, marginTop: '0.25rem', borderTop: '1px solid #253545', paddingTop: '0.4rem' }}>
                            <strong style={{ color: '#f7b731' }}>Gradual Spectrum:</strong>{' '}
                            Fibrosis stages (F0–F4) overlap continuously. FCM assigns probabilistic membership rather than a hard category.
                        </p>
                    </div>
                )}

                {/* Mini-map of modified areas */}
                {deltaMap && (
                    <div className="report-card" style={{ border: '1px solid #4ecdc4' }}>
                        <p className="report-label">Modified Areas Map</p>
                        <div className="delta-minimap-panel">
                            <img src={deltaMap} alt="Modified areas" draggable="false" />
                            <div className="delta-minimap-legend">
                                <span><span className="legend-swatch legend-green" /> Modified</span>
                                <span><span className="legend-swatch legend-white" /> Original</span>
                            </div>
                        </div>
                    </div>
                )}

            </aside>

            {showResetConfirm && (
                <div className="confirm-overlay" onClick={handleConfirmNo}>
                    <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
                        <p className="confirm-title">Reset Adjustments?</p>
                        <p className="confirm-text">
                            You have made fine area adjustments that will be
                            reset to a general threshold. This cannot be undone.
                        </p>
                        <div className="confirm-hint">Press Esc to cancel</div>
                        <div className="confirm-actions">
                            <button className="confirm-btn confirm-btn-cancel" onClick={handleConfirmNo}>Cancel</button>
                            <button className="confirm-btn confirm-btn-proceed" onClick={handleConfirmYes}>Reset</button>
                        </div>
                    </div>
                </div>
            )}

            {showHelp && (
                <div className="confirm-overlay" onClick={() => setShowHelp(false)}>
                    <div className="help-modal" onClick={(e) => e.stopPropagation()}>
                        <button className="help-close" onClick={() => setShowHelp(false)}>✕</button>
                        <h2 className="help-title">How to Use AI-Fibrosis</h2>

                        <div className="help-section">
                            <h3>Analysis</h3>
                            <p>Upload an image (SVS, TIF, JPG, PNG, BMP) by clicking or dragging into the left panel. The AI pipeline will automatically extract collagen fibers, compute a fibrosis extent percentage, and classify the disease stage via VGG16 + Fuzzy C-Means.</p>
                        </div>

                        <div className="help-section">
                            <h3>Baseline Threshold</h3>
                            <p>After analysis, the right panel provides a <strong>Baseline Threshold</strong> slider. This controls the global sensitivity for detecting fibrosis. Once you set a baseline you are happy with, <strong>stick to fine area adjustments</strong> rather than changing the baseline again. Changing the baseline resets all area edits. Use <em>Reset to baseline AI estimate</em> only for a full reset.</p>
                        </div>

                        <div className="help-section">
                            <h3>Magnifying Glass & Fine Adjustments</h3>
                            <p>Hover over the original image to activate the magnifying glass. Use arrow keys to zoom and adjust the threshold for just the area under observation:</p>
                            <ul className="help-bindings">
                                <li><kbd>▲</kbd> <kbd>▼</kbd> Zoom in / out</li>
                                <li><kbd>◀</kbd> <kbd>▶</kbd> Decrease / increase area threshold</li>
                                <li><kbd>Ctrl+R</kbd> Reset observed area to original threshold</li>
                                <li><kbd>Ctrl+Z</kbd> Undo all increments made to the last modified area</li>
                                <li><kbd>Esc</kbd> Close overlays</li>
                            </ul>
                        </div>

                        <div className="help-section">
                            <h3>Mini-Map</h3>
                            <p>When area adjustments are made, a <strong>Modified Areas Map</strong> appears in the diagnosis panel showing which regions have been modified (green) vs. the original AI threshold (white). The map updates live as you undo or reset regions.</p>
                        </div>

                        <div className="help-section">
                            <h3>Exports</h3>
                            <p><strong>Download CSV</strong> exports the diagnosis result. <strong>Save Mask</strong> downloads the current fibrosis mask image (including any threshold adjustments).</p>
                        </div>

                        <div className="help-esc">Press Esc to close</div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ImageSubmission;