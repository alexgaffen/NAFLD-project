import { useState, useRef, useCallback, useEffect } from "react";
import { createPortal } from "react-dom";

const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

// ── Auth: transparent access-token refresh ──────────────────────────
// The backend issues short-lived (30 min) access tokens and longer-lived
// (7 day) refresh tokens. Previously the app would log out on the first
// 401 after expiry; now we transparently call /refresh and retry.

const _decodeJwt = (token) => {
    try {
        const payload = token.split('.')[1];
        const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
        return JSON.parse(json);
    } catch { return null; }
};

const _msUntilExpiry = (token) => {
    const p = _decodeJwt(token);
    if (!p || !p.exp) return 0;
    return p.exp * 1000 - Date.now();
};

// In-flight refresh promise so concurrent requests share one /refresh call.
let _refreshPromise = null;

const refreshAccessToken = () => {
    if (_refreshPromise) return _refreshPromise;
    const rt = sessionStorage.getItem("refresh_token");
    if (!rt) return Promise.resolve(null);
    _refreshPromise = fetch(`${API_BASE}/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: rt }),
    })
        .then(async (r) => {
            if (!r.ok) return null;
            const data = await r.json().catch(() => null);
            if (data && data.access_token) {
                sessionStorage.setItem("access_token", data.access_token);
                if (data.username) sessionStorage.setItem("username", data.username);
                return data.access_token;
            }
            return null;
        })
        .catch(() => null)
        .finally(() => { _refreshPromise = null; });
    return _refreshPromise;
};

/** Return a usable access token, refreshing first if it expires within minMs. */
const ensureFreshToken = async (minMs = 60000) => {
    const t = sessionStorage.getItem("access_token");
    if (t && _msUntilExpiry(t) > minMs) return t;
    const refreshed = await refreshAccessToken();
    return refreshed || t || null;
};

const _sessionExpired = () => {
    sessionStorage.removeItem("access_token");
    sessionStorage.removeItem("refresh_token");
    sessionStorage.removeItem("username");
    window.location.reload();
};

/** Authenticated fetch with one-shot refresh + retry on 401. */
const apiFetch = async (url, options = {}) => {
    const doFetch = (token) => {
        const headers = { ...(options.headers || {}) };
        if (token) headers.Authorization = `Bearer ${token}`;
        return fetch(url, { ...options, headers });
    };
    let token = await ensureFreshToken();
    let res = await doFetch(token);
    if (res.status === 401) {
        const newToken = await refreshAccessToken();
        if (!newToken) { _sessionExpired(); throw new Error('Session expired'); }
        res = await doFetch(newToken);
        if (res.status === 401) { _sessionExpired(); throw new Error('Session expired'); }
    }
    return res;
};

/** Build an authenticated SSE URL (refreshing token first if needed). */
const buildSseUrl = async (path) => {
    // Ensure plenty of margin (2 min) since SSE streams are long-lived.
    const token = await ensureFreshToken(120000);
    if (!token) { _sessionExpired(); throw new Error('Session expired'); }
    const sep = path.includes('?') ? '&' : '?';
    return `${API_BASE}${path}${sep}token=${encodeURIComponent(token)}`;
};

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

    // Classification from refined mask (separate from initial analysis)
    const [classificationResult, setClassificationResult] = useState(null);
    const [isClassifying, setIsClassifying] = useState(false);
    const [classifyTileGrid, setClassifyTileGrid] = useState(null);
    const [classifyProgress, setClassifyProgress] = useState(null);
    const [worstPatchCoords, setWorstPatchCoords] = useState(null);
    const [showWorstPatches, setShowWorstPatches] = useState(false);
    const [maskChangedSinceClassify, setMaskChangedSinceClassify] = useState(false);
    const [filteredImgContentRect, setFilteredImgContentRect] = useState(null);
    const classifyGridInfo = useRef(null); // { rows, cols } from classification

    // Excluded-pixels overlay (shows what is removed from the extent denominator)
    const [showExcluded, setShowExcluded] = useState(false);
    const [excludedOverlay, setExcludedOverlay] = useState(null);
    const [isLoadingExcluded, setIsLoadingExcluded] = useState(false);

    // Area inspection (Q key while magnifier active): kicks off a fast
    // extent + classify on the region under the lens. Auto-cancels
    // when the cursor leaves the locked region.
    const [areaExtent, setAreaExtent] = useState(null);     // { state, ratio }
    const [areaClassify, setAreaClassify] = useState(null); // { state, scores, label }
    const areaInspectRef = useRef({ region: null, abort: null, panelPos: null });

    const cancelAreaInspect = useCallback(() => {
        const cur = areaInspectRef.current;
        if (cur.abort) { try { cur.abort.abort(); } catch (e) {} }
        areaInspectRef.current = { region: null, abort: null, panelPos: null };
        setAreaExtent(null);
        setAreaClassify(null);
    }, []);

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

    // Compute where the filtered image renders inside its panel
    const computeFilteredImgRect = useCallback(() => {
        const img = filteredImgRef.current;
        if (!img || !img.naturalWidth) return;
        const container = img.parentElement;
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const nw = img.naturalWidth;
        const nh = img.naturalHeight;
        const scale = Math.min(cw / nw, ch / nh);
        const rw = nw * scale;
        const rh = nh * scale;
        setFilteredImgContentRect({
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

    // Recompute filtered image rect on resize
    useEffect(() => {
        const img = filteredImgRef.current;
        if (!img) return;
        const ro = new ResizeObserver(computeFilteredImgRect);
        ro.observe(img.parentElement);
        return () => ro.disconnect();
    }, [computeFilteredImgRect, displayedResult, adjustedMask]);

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
                const res = await apiFetch(
                    `${API_BASE}/rethreshold/${encodeURIComponent(uploadedFilename)}?threshold=${value}`
                );
                if (res.ok) {
                    const data = await res.json();
                    setAdjustedRatio(data.fibrosis_ratio);
                    setAdjustedMask(data.filtered_image);
                    setHasLocalEdits(false);
                    setDeltaMap(null);
                    setMaskChangedSinceClassify(true);
                    setShowWorstPatches(false);
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
        const newPos = { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
        // While an area inspection is locked, freeze the lens at the pinned
        // position. Only cancel (and resume tracking) once the cursor
        // wanders well outside the lens footprint.
        const inspect = areaInspectRef.current;
        if (inspect.panelPos) {
            const dx = Math.abs(newPos.x - inspect.panelPos.x);
            const dy = Math.abs(newPos.y - inspect.panelPos.y);
            if (dx > 0.08 || dy > 0.08) {
                cancelAreaInspect();
                setMagnifier(newPos);
            }
            return;
        }
        setMagnifier(newPos);
    }, [cancelAreaInspect]);
    const handlePanelMouseLeave = useCallback(() => {
        cancelAreaInspect();
        setMagnifier(null);
    }, [cancelAreaInspect]);

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
                const res = await apiFetch(
                    `${API_BASE}/rethreshold-area/${encodeURIComponent(uploadedFilename)}?${params}`
                );
                if (res.ok) {
                    const data = await res.json();
                    setAdjustedRatio(data.fibrosis_ratio);
                    setAdjustedMask(data.filtered_image);
                    setHasLocalEdits(true);
                    if (data.delta_map) setDeltaMap(data.delta_map);
                    setMaskChangedSinceClassify(true);
                    setShowWorstPatches(false);
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
            const res = await apiFetch(
                `${API_BASE}/reset-area/${encodeURIComponent(uploadedFilename)}?${params}`
            );
            if (res.ok) {
                const data = await res.json();
                setAdjustedRatio(data.fibrosis_ratio);
                setAdjustedMask(data.filtered_image);
                setHasLocalEdits(data.has_local_edits);
                setDeltaMap(data.delta_map);
                setMaskChangedSinceClassify(true);
                setShowWorstPatches(false);
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
            const res = await apiFetch(
                `${API_BASE}/undo-area/${encodeURIComponent(uploadedFilename)}`
            );
            if (res.ok) {
                const data = await res.json();
                setAdjustedRatio(data.fibrosis_ratio);
                setAdjustedMask(data.filtered_image);
                setHasLocalEdits(data.has_local_edits);
                setDeltaMap(data.delta_map);
                setMaskChangedSinceClassify(true);
                setShowWorstPatches(false);
            }
        } catch (e) {
            console.error('Undo area error:', e);
        } finally {
            setIsRethresholding(false);
        }
    }, [uploadedFilename]);

    // Classify from refined mask (SSE streaming with tile progress)
    const handleClassifyMask = useCallback(async () => {
        if (!uploadedFilename) return;
        setIsClassifying(true);
        setClassificationResult(null);
        setClassifyTileGrid(null);
        setClassifyProgress(null);
        setWorstPatchCoords(null);
        setShowWorstPatches(false);
        classifyGridInfo.current = null;

        const maxRetries = 3;
        let lastError = null;

        for (let attempt = 0; attempt < maxRetries; attempt++) {
            if (attempt > 0) {
                await new Promise(r => setTimeout(r, 1000));
                setClassifyTileGrid(null);
                setClassifyProgress(null);
            }
            try {
                const result = await new Promise((resolve, reject) => {
                    let settled = false;
                    let evtSource = null;
                    buildSseUrl(`/classify-mask/${encodeURIComponent(uploadedFilename)}`)
                        .then((url) => {
                            evtSource = new EventSource(url);
                            evtSource.onmessage = (event) => {
                                try {
                                    const msg = JSON.parse(event.data);
                                    if (msg.type === 'progress') {
                                        setClassifyProgress({ current: msg.current, total: msg.total, tissue_patches: msg.tissue_patches });
                                        if (msg.grid_rows !== undefined) {
                                            setClassifyTileGrid(prev => {
                                                const r = msg.grid_rows, c = msg.grid_cols;
                                                const tiles = prev && prev.tiles.length === r * c ? [...prev.tiles] : new Array(r * c).fill(0);
                                                tiles[msg.tile_row * c + msg.tile_col] = msg.is_tissue ? 2 : 1;
                                                return { rows: r, cols: c, tiles };
                                            });
                                        }
                                    } else if (msg.type === 'result') {
                                        settled = true;
                                        evtSource.close();
                                        resolve(msg.data);
                                    }
                                } catch (e) {
                                    if (!settled) { settled = true; evtSource.close(); reject(e); }
                                }
                            };
                            evtSource.onerror = () => {
                                if (!settled) { settled = true; evtSource.close(); reject(new Error('Connection lost')); }
                            };
                        })
                        .catch((e) => { if (!settled) { settled = true; reject(e); } });
                });
                setClassificationResult(result);
                if (result && result.worst_patches) {
                    setWorstPatchCoords(result.worst_patches);
                    classifyGridInfo.current = {
                        rows: result.grid_rows, cols: result.grid_cols,
                        imgH: result.img_h, imgW: result.img_w,
                        patchSize: result.patch_size || 512,
                    };
                }
                setMaskChangedSinceClassify(false);
                lastError = null;
                break;
            } catch (e) {
                lastError = e;
                console.warn(`Classify attempt ${attempt + 1}/${maxRetries} failed:`, e.message);
            }
        }
        if (lastError) {
            console.error('Classify mask error after retries:', lastError);
            setErrorMessage('Classification failed or connection lost.');
        }
        setIsClassifying(false);
        setClassifyTileGrid(null);
        setClassifyProgress(null);
    }, [uploadedFilename]);

    // Toggle the green "excluded pixels" overlay on the original image.
    // The overlay paints exactly the pixels that were dropped from the
    // extent denominator (too dark or too white to be tissue).
    const handleToggleExcluded = useCallback(async () => {
        if (!uploadedFilename) return;
        if (showExcluded) {
            setShowExcluded(false);
            return;
        }
        if (excludedOverlay) {
            setShowExcluded(true);
            return;
        }
        setIsLoadingExcluded(true);
        try {
            const res = await apiFetch(`${API_BASE}/preview-excluded/${encodeURIComponent(uploadedFilename)}`);
            if (res.ok) {
                const data = await res.json();
                setExcludedOverlay(data.overlay);
                setShowExcluded(true);
            }
        } catch (e) {
            console.error('Excluded-mask fetch error:', e);
        } finally {
            setIsLoadingExcluded(false);
        }
    }, [uploadedFilename, showExcluded, excludedOverlay]);

    // Press Q with magnifier active to run a fast extent + classify on the
    // area under the lens. Both calls share an AbortController so a mouse
    // move (handled in handlePanelMouseMove) cancels them in flight.
    const handleAreaInspect = useCallback(() => {
        if (!uploadedFilename || !magnifier) return;
        const region = getMagnifierRegion();
        if (!region) return;
        // Re-entry: cancel any in-flight inspection first
        const prev = areaInspectRef.current;
        if (prev.abort) { try { prev.abort.abort(); } catch (e) {} }

        const ctrl = new AbortController();
        areaInspectRef.current = {
            panelPos: { x: magnifier.x, y: magnifier.y }, // pin the lens in panel space
            region: region,
            abort: ctrl,
        };
        setAreaExtent({ state: 'loading' });
        setAreaClassify({ state: 'loading' });

        const params = new URLSearchParams({
            x1: region.x1, y1: region.y1, x2: region.x2, y2: region.y2,
        });

        // Extent — fast, runs against cached red_stain.
        apiFetch(`${API_BASE}/analyze-area/${encodeURIComponent(uploadedFilename)}?${params}`,
            { signal: ctrl.signal })
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
            .then(data => {
                if (areaInspectRef.current.abort !== ctrl) return; // stale
                setAreaExtent({ state: 'done', ratio: data.fibrosis_ratio });
            })
            .catch(e => {
                if (e.name === 'AbortError') return;
                if (areaInspectRef.current.abort !== ctrl) return;
                setAreaExtent({ state: 'error' });
            });

        // Classify — VGG16 + PCA + FCM on the cropped mask region.
        apiFetch(`${API_BASE}/classify-mask-area/${encodeURIComponent(uploadedFilename)}?${params}`,
            { signal: ctrl.signal })
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
            .then(data => {
                if (areaInspectRef.current.abort !== ctrl) return;
                if (data.status === 'success') {
                    setAreaClassify({
                        state: 'done',
                        scores: data.membership_scores,
                        label: data.cluster_label,
                    });
                } else {
                    setAreaClassify({ state: 'background' });
                }
            })
            .catch(e => {
                if (e.name === 'AbortError') return;
                if (areaInspectRef.current.abort !== ctrl) return;
                setAreaClassify({ state: 'error' });
            });
    }, [uploadedFilename, magnifier, getMagnifierRegion]);

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
            // Q: instant area inspection (extent + classify) of region under lens
            if (e.key === 'q' || e.key === 'Q') {
                e.preventDefault();
                handleAreaInspect();
                return;
            }
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
    }, [magnifier, autoThreshold, getMagnifierRegion, handleLocalDeltaChange, handleResetArea, handleUndoArea, showHelp, showResetConfirm, handleConfirmNo, handleAreaInspect]);

    const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB per chunk
    const LARGE_FILE_THRESHOLD = 10 * 1024 * 1024; // 10 MB -- use chunking above this

    // ---- Chunked upload for large files (SVS, big TIF) ----
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

            const res = await apiFetch(`${API_BASE}/largefile`, { method: 'POST', body: form });
            if (!res.ok) throw new Error(`Chunk ${i + 1}/${totalChunks} failed: ${await res.text()}`);

            const data = await res.json();
            setUploadProgress(Math.round(((i + 1) / totalChunks) * 100));

            // Last chunk returns the filename
            if (data.filename) serverFilename = data.filename;
        }

        if (!serverFilename) throw new Error('Chunked upload completed but no filename returned.');
        return serverFilename;
    };

    // ---- Simple upload for small files ----
    const uploadSimple = async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await apiFetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
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
        setClassificationResult(null);
        setClassifyTileGrid(null);
        setClassifyProgress(null);
        setWorstPatchCoords(null);
        setShowWorstPatches(false);
        setMaskChangedSinceClassify(false);
        setFilteredImgContentRect(null);
        classifyGridInfo.current = null;
        setUploadedFilename("");
        setUploadProgress(0);
        setAutoThreshold(null);
        setUserThreshold(null);
        setAdjustedRatio(null);
        setAdjustedMask(null);
        setMagnifier(null);
        setHasLocalEdits(false);
        setShowResetConfirm(false);
        setDeltaMap(null);
        setShowExcluded(false);
        setExcludedOverlay(null);

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
                const previewResponse = await apiFetch(`${API_BASE}/preview/${encodeURIComponent(filename)}`);
                if (previewResponse.ok) {
                    setPreviewResult(await previewResponse.json());
                }

                // Stream patch progress via SSE
                setIsUploading(false);
                setIsAnalyzing(true);
                setPatchProgress(null);

                const analyzeResult = await new Promise((resolve, reject) => {
                    let settled = false;
                    let evtSource = null;
                    buildSseUrl(`/analyze-stream/${encodeURIComponent(filename)}`)
                        .then((url) => {
                            evtSource = new EventSource(url);
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
                                    } else if (msg.type === 'result') {
                                        settled = true;
                                        evtSource.close();
                                        resolve(msg.data);
                                    }
                                } catch (e) {
                                    if (!settled) { settled = true; evtSource.close(); reject(e); }
                                }
                            };
                            evtSource.onerror = () => {
                                if (!settled) { settled = true; evtSource.close(); reject(new Error('Analysis stream connection lost')); }
                            };
                        })
                        .catch((e) => { if (!settled) { settled = true; reject(e); } });
                });
                setAnalysisResult(analyzeResult);
            } else {
                // Small images (JPG/PNG/BMP): skip preview, single analysis call returns everything
                setIsUploading(false);
                setIsAnalyzing(true);
                setPatchProgress(null);

                const analyzeResponse = await apiFetch(`${API_BASE}/analyze/${encodeURIComponent(filename)}`);
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

    const handleDownloadCsv = () => {
        const extent = adjustedRatio !== null ? adjustedRatio : fibrosisRatio;
        const scores = classificationResult?.membership_scores;
        if (extent === undefined || !scores) { setErrorMessage("Complete a full diagnosis (including clustering) before downloading CSV."); return; }
        const headers = ['image_name', 'extent_percentage', 'None (F0)', 'Periportal (F1)', 'Bridging (F3)', 'Cirrhosis (F4)'];
        const values = [
            image?.name || uploadedFilename,
            `${Number(extent).toFixed(2)}%`,
            `${((scores.None || 0) * 100).toFixed(0)}%`,
            `${((scores.Perisinusoidal || 0) * 100).toFixed(0)}%`,
            `${((scores.Bridging || 0) * 100).toFixed(0)}%`,
            `${((scores.Cirrosis || 0) * 100).toFixed(0)}%`
        ];
        const csvContent = [headers.join(','), values.join(',')].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        const baseName = image?.name ? image.name.replace(/\.[^.]+$/, '') : 'fibrosis';
        link.href = url;
        link.setAttribute('download', `${baseName}_results.csv`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
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
        const cx = 170, cy = 150, R = 80;
        const N = 4;
        const cats = [
            { key: 'None',            label: 'None',            sub: 'F0' },
            { key: 'Perisinusoidal',  label: 'Periportal',      sub: 'F1' },
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
        // Find the dominant (highest-membership) category
        const maxV = Math.max(...pts.map(p => p.v));
        const dominantIdx = pts.findIndex(p => p.v === maxV);
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
        const dominant = pts[dominantIdx];
        return (
            <>
                {/* Dominant-class banner */}
                {dominant && (
                    <div className="dominant-class-banner">
                        <span className="dominant-class-tag">PRIMARY CLASSIFICATION</span>
                        <div className="dominant-class-row">
                            <span className="dominant-class-label">{dominant.label}</span>
                            <span className="dominant-class-stage">({dominant.sub})</span>
                            <span className="dominant-class-pct">{(dominant.v * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                )}
                <svg viewBox="0 0 380 310" style={{ width: '100%', maxWidth: 340, display: 'block', margin: '0.5rem auto' }}>
                    <defs>
                        <radialGradient id="dominantGlow" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="#f7b731" stopOpacity="0.55" />
                            <stop offset="100%" stopColor="#f7b731" stopOpacity="0" />
                        </radialGradient>
                    </defs>
                    {webLines.map((w, i) => (
                        <polygon key={`web${i}`} points={w}
                            fill="none" stroke="#253545" strokeWidth={0.75}
                            strokeDasharray={rings[i] < 1 ? '4 4' : 'none'} />
                    ))}
                    {pts.map((p, i) => (
                        <line key={`ax${i}`} x1={cx} y1={cy} x2={p.ex} y2={p.ey}
                            stroke={i === dominantIdx ? '#f7b731' : '#4ecdc4'}
                            strokeWidth={i === dominantIdx ? 2.5 : 1.5}
                            opacity={i === dominantIdx ? 0.85 : 0.45} />
                    ))}
                    <polygon points={poly} fill="rgba(78,205,196,0.22)" stroke="#4ecdc4" strokeWidth={2} />
                    {/* Dominant-vertex glow */}
                    {dominant && dominant.v > 0 && (
                        <circle cx={dominant.x} cy={dominant.y} r={18}
                            fill="url(#dominantGlow)" />
                    )}
                    {pts.map((p, i) => (
                        <circle key={`dot${i}`} cx={p.x} cy={p.y}
                            r={i === dominantIdx ? 7 : 4.5}
                            fill={i === dominantIdx ? '#f7b731' : '#0f1923'}
                            stroke={i === dominantIdx ? '#f7b731' : '#4ecdc4'}
                            strokeWidth={i === dominantIdx ? 2.5 : 2} />
                    ))}
                    {pts.map((p) => {
                        const a = p.angle;
                        const cos = Math.cos(a), sin = Math.sin(a);
                        const isTop = sin < -0.3, isBot = sin > 0.3, isLeft = cos < -0.3, isRight = cos > 0.3;
                        const anchor = isRight ? 'start' : isLeft ? 'end' : 'middle';
                        const dx = isRight ? 14 : isLeft ? -14 : 0;
                        const dy1 = isTop ? -22 : isBot ? 18 : -6;
                        const dy2 = dy1 + 12;
                        const isDom = p.idx === dominantIdx;
                        return (
                            <g key={`lbl${p.idx}`}>
                                <text x={p.ex + dx} y={p.ey + dy1}
                                    fill={isDom ? '#f7b731' : '#fff'}
                                    fontSize={isDom ? '17' : '15'} fontWeight="700"
                                    textAnchor={anchor}>{`${(p.v * 100).toFixed(0)}%`}</text>
                                <text x={p.ex + dx} y={p.ey + dy2}
                                    fill={isDom ? '#f7b731' : '#8a9bae'}
                                    fontSize="12"
                                    fontWeight={isDom ? '700' : '400'}
                                    textAnchor={anchor}>{p.label}</text>
                            </g>
                        );
                    })}
                </svg>
            </>
        );
    };

    // The mask to display: use adjusted mask if user moved slider, else the original
    const displayedMaskSrc = adjustedMask || displayedResult?.filtered_image;

    // (Tile overlay during analyze removed — the analyze pass no longer
    // does per-tile work, so there's nothing to visualise. The classify
    // step still shows its tile scan on the right panel.)

    // Magnifier rendering helper
    const zoomFraction = (magnifierZoom - MIN_ZOOM) / (MAX_ZOOM - MIN_ZOOM);
    const renderMagnifier = (imgRef, side = 'left') => {
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

        // Convert panel-local coords to viewport coords so we can portal
        // the magnifier out of any clipping ancestors (e.g. .img-panel
        // overflow:hidden) and have it float over the rest of the UI.
        const panelRect = container.getBoundingClientRect();
        const vpLeft = panelRect.left + posX - MAGNIFIER_SIZE / 2;
        const vpTop = panelRect.top + posY - MAGNIFIER_SIZE / 2;

        // Inspection attachment for this magnifier
        let attachment = null;
        if (side === 'left' && areaExtent) {
            attachment = (
                <div className="mag-attach mag-attach-extent">
                    <span className="mag-attach-label">Area Extent</span>
                    {areaExtent.state === 'loading' && <span className="mag-attach-spinner" />}
                    {areaExtent.state === 'done' && (
                        <>
                            <span className="mag-attach-value">{areaExtent.ratio.toFixed(2)}%</span>
                            <div className="mag-attach-bar">
                                <div className="mag-attach-bar-fill" style={{ width: `${Math.min(areaExtent.ratio, 100)}%` }} />
                            </div>
                        </>
                    )}
                    {areaExtent.state === 'error' && <span className="mag-attach-err">err</span>}
                </div>
            );
        } else if (side === 'right' && areaClassify) {
            attachment = (
                <div className="mag-attach mag-attach-classify">
                    <span className="mag-attach-label">Area Stage</span>
                    {areaClassify.state === 'loading' && <span className="mag-attach-spinner" />}
                    {areaClassify.state === 'done' && areaClassify.scores && (
                        <>
                            <span className="mag-attach-value mag-attach-label-text">{areaClassify.label}</span>
                            <div className="mag-attach-stages">
                                {Object.entries(areaClassify.scores).map(([k, v]) => (
                                    <div key={k} className="mag-stage-row">
                                        <span className="mag-stage-name">{k.slice(0, 4)}</span>
                                        <div className="mag-stage-bar"><div className="mag-stage-fill" style={{ width: `${v * 100}%` }} /></div>
                                        <span className="mag-stage-pct">{Math.round(v * 100)}</span>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}
                    {areaClassify.state === 'background' && <span className="mag-attach-err">background</span>}
                    {areaClassify.state === 'error' && <span className="mag-attach-err">err</span>}
                </div>
            );
        }

        // "Press Q" hint only when no inspection is active
        const showQHint = !areaExtent && !areaClassify && analysisResult;

        return createPortal((
            <div className="magnifier-group" style={{
                position: 'fixed',
                left: vpLeft,
                top: vpTop,
                zIndex: 9999,
            }}>
                <div className="mag-main-col">
                    <div className="mag-top-row">
                        <div className="magnifier-lens" style={{
                            width: MAGNIFIER_SIZE,
                            height: MAGNIFIER_SIZE,
                            backgroundImage: `url(${img.src})`,
                            backgroundSize: `${bgW}px ${bgH}px`,
                            backgroundPosition: `${bgX}px ${bgY}px`,
                        }}>
                            {showQHint && <span className="mag-q-hint"><kbd>Q</kbd> inspect</span>}
                        </div>
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
                    {attachment}
                </div>
            </div>
        ), document.body);
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
                                    <span className="loading-hourglass" />
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
                        {magnifier && displayedResult?.original_image && renderMagnifier(originalImgRef, 'left')}
                        {magnifier && displayedResult?.original_image && <div className="magnifier-dim" />}

                        {/* Filename & change message at bottom of frame */}
                        {image && (
                            <div className="img-panel-footer">
                                <span className="img-panel-filename">{image.name}</span>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                    {!isBusy && <span className="img-panel-change" onClick={handleClick}>Click or drop to change</span>}
                                    {displayedResult?.original_image && uploadedFilename && !isBusy && (
                                        <button
                                            className={`excluded-toggle-btn${showExcluded ? ' active' : ''}`}
                                            onClick={(e) => { e.stopPropagation(); handleToggleExcluded(); }}
                                            title={showExcluded ? 'Hide excluded pixels' : 'Highlight pixels excluded from extent %'}
                                            disabled={isLoadingExcluded}
                                        >
                                            {isLoadingExcluded ? '…' : showExcluded ? '● what was excluded?' : '○ what was excluded?'}
                                        </button>
                                    )}
                                    {isAnalyzing && <button className="cancel-diagnosis-btn" onClick={() => window.location.reload()}>× Cancel Diagnosis</button>}
                                </div>
                            </div>
                        )}

                        {/* Tile grid overlay removed — analyze no longer scans tile-by-tile. */}

                        {/* Excluded-pixels overlay (green = not counted in extent denominator) */}
                        {showExcluded && excludedOverlay && imgContentRect && (
                            <img
                                alt="Excluded pixels"
                                src={excludedOverlay}
                                draggable="false"
                                style={{
                                    position: 'absolute',
                                    left: imgContentRect.left,
                                    top: imgContentRect.top,
                                    width: imgContentRect.width,
                                    height: imgContentRect.height,
                                    pointerEvents: 'none',
                                    imageRendering: 'pixelated',
                                }}
                            />
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
                        <span className="img-label accent">fibrosisai Mask</span>
                        {displayedMaskSrc ? (
                            <img ref={filteredImgRef} onLoad={computeFilteredImgRect} alt="Fibrosis mask" src={displayedMaskSrc} className="preview-image" draggable="false" onDragStart={(e) => e.preventDefault()} />
                        ) : (
                            <div className="placeholder-text">Fibrosis mask appears after analysis</div>
                        )}
                        {magnifier && displayedMaskSrc && renderMagnifier(filteredImgRef, 'right')}
                        {magnifier && displayedMaskSrc && <div className="magnifier-dim" />}
                        {displayedMaskSrc && (
                            !analysisResult ? (
                                <div className="download-mask-btn" style={{ color: '#a855f7', borderColor: '#a855f7', pointerEvents: 'none', cursor: 'default', background: 'transparent' }}>
                                    preview only
                                </div>
                            ) : (
                                <button className="download-mask-btn" onClick={handleDownloadMask} title="Download fibrosis mask">
                                    ↓ Save Mask
                                </button>
                            )
                        )}

                        {/* Classify scan tile overlay — on filtered image during classification */}
                        {classifyTileGrid && isClassifying && filteredImgContentRect && (
                            <div className="tile-overlay" style={{
                                left: filteredImgContentRect.left,
                                top: filteredImgContentRect.top,
                                width: filteredImgContentRect.width,
                                height: filteredImgContentRect.height,
                                gridTemplateColumns: `repeat(${classifyTileGrid.cols}, 1fr)`,
                                gridTemplateRows: `repeat(${classifyTileGrid.rows}, 1fr)`,
                            }}>
                                {classifyTileGrid.tiles.map((s, i) => (
                                    <div key={i} className={`tile-cell ${s === 2 ? 'tile-tissue' : s === 1 ? 'tile-bg' : 'tile-pending'}`} />
                                ))}
                            </div>
                        )}

                        {/* Classify progress bar */}
                        {isClassifying && classifyProgress && (
                            <div className="img-progress-bar-track">
                                <div
                                    className="img-progress-bar-fill"
                                    style={{ width: `${Math.round((classifyProgress.current / classifyProgress.total) * 100)}%` }}
                                />
                            </div>
                        )}

                        {/* Red outlines for worst patches — pixel-accurate positioning */}
                        {showWorstPatches && worstPatchCoords && filteredImgContentRect && classifyGridInfo.current && (
                            <div className="worst-patches-overlay" style={{
                                left: filteredImgContentRect.left,
                                top: filteredImgContentRect.top,
                                width: filteredImgContentRect.width,
                                height: filteredImgContentRect.height,
                            }}>
                                {worstPatchCoords.map((p, i) => {
                                    const gi = classifyGridInfo.current;
                                    // Map pixel coords to display coords using actual image dimensions
                                    const scaleX = filteredImgContentRect.width / gi.imgW;
                                    const scaleY = filteredImgContentRect.height / gi.imgH;
                                    const patchW = Math.min(gi.patchSize, gi.imgW - p.px) * scaleX;
                                    const patchH = Math.min(gi.patchSize, gi.imgH - p.py) * scaleY;
                                    return (
                                        <div key={i} className="worst-patch-rect" style={{
                                            left: p.px * scaleX,
                                            top: p.py * scaleY,
                                            width: patchW,
                                            height: patchH,
                                        }}>
                                            <span className="worst-patch-rank">{i + 1}</span>
                                        </div>
                                    );
                                })}
                            </div>
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
                        disabled={!classificationResult?.membership_scores || isUploading || isAnalyzing}
                    >
                        ↓ Download CSV
                    </button>
                </div>

                {/* Extent calculation in progress — animated indeterminate indicator */}
                {isAnalyzing && (
                    <div className="report-card" style={{ borderColor: '#4ecdc4' }}>
                        <p className="report-label" style={{ color: '#4ecdc4', fontWeight: 600 }}>
                            Extent Calculation in Progress<span className="extent-dots" />
                        </p>
                        <div className="extent-scan-track">
                            <div className="extent-scan-fill" />
                        </div>
                        <p style={{ fontSize: '0.72rem', color: '#fff', marginTop: '0.3rem', marginBottom: 0 }}>
                            Deconvolving PSR stain · sweeping adaptive threshold · isolating collagen pixels<span className="extent-dots" />
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
                                min={Math.max(autoThreshold - 5, 0)}
                                max={autoThreshold + 5}
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
                                        setMaskChangedSinceClassify(true);
                                        setShowWorstPatches(false);
                                        apiFetch(`${API_BASE}/rethreshold/${encodeURIComponent(uploadedFilename)}?threshold=${autoThreshold}`)
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

                {/* FCM Membership Radar Chart — 4 categories + spectrum note */}
                {analysisResult && !isAnalyzing && (
                    <div className="report-card" style={{ border: '1px solid #4ecdc4' }}>
                        <p className="report-label">Disease Classification</p>

                        {/* Always show hint to fine-tune before diagnosing */}
                        <div className="classify-prompt">
                            <p className="classify-hint">
                                Fine-tune the fibrosis mask using the baseline threshold slider and magnifying glass area adjustments. Once the mask accurately represents the fibrosis, click below to classify.
                            </p>
                        </div>

                        {!classificationResult && !isClassifying && (
                            <button className="classify-btn" onClick={handleClassifyMask}>
                                Diagnose
                            </button>
                        )}

                        {isClassifying && (
                            <div className="classify-loading">
                                <span className="classify-spinner" />
                                <span>Classifying refined mask{classifyProgress ? ` (${classifyProgress.current}/${classifyProgress.total})` : ''}...</span>
                            </div>
                        )}

                        {classificationResult && classificationResult.status === 'success' && (
                            <>
                                {renderRadarChart(classificationResult)}
                                {classificationResult.top_n_used && (
                                    <p style={{ fontSize: '0.72rem', color: '#8a9bae', marginTop: '0.25rem' }}>
                                        Based on the top {classificationResult.top_n_used} most severe tiles out of {classificationResult.patch_count} classified
                                    </p>
                                )}
                                <p style={{ fontSize: '0.78rem', color: '#fff', lineHeight: 1.5, marginTop: '0.25rem', borderTop: '1px solid #253545', paddingTop: '0.4rem' }}>
                                    <strong style={{ color: '#f7b731' }}>Gradual Spectrum:</strong>{' '}
                                    Fibrosis categories overlap continuously. FCM assigns probabilistic membership across 4 clusters rather than a hard category.
                                </p>
                                <div className="classify-actions-row">
                                    <button
                                        className="reclassify-btn"
                                        onClick={handleClassifyMask}
                                        disabled={!maskChangedSinceClassify}
                                        title={!maskChangedSinceClassify ? 'Make threshold adjustments first' : 'Re-run classification on updated mask'}
                                    >
                                        Re-diagnose
                                    </button>
                                    {worstPatchCoords && worstPatchCoords.length > 0 && (
                                        <button
                                            className="analyze-patches-btn"
                                            onClick={() => setShowWorstPatches(prev => !prev)}
                                        >
                                            {showWorstPatches ? 'Hide Patch Results' : 'Analyze Patch Results'}
                                        </button>
                                    )}
                                </div>
                            </>
                        )}
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
                        <button className="help-close" onClick={() => setShowHelp(false)}>&times;</button>
                        <h2 className="help-title">How to Use fibrosisai</h2>

                        <div className="help-section">
                            <h3>Analysis</h3>
                            <p>Upload an image (SVS, TIF, JPG, PNG, BMP) by clicking or dragging into the left panel. The AI pipeline will automatically extract collagen fibers via colour deconvolution and compute a fibrosis extent percentage. The original image appears on the left and the fibrosis mask on the right.</p>
                        </div>

                        <div className="help-section">
                            <h3>Baseline Threshold</h3>
                            <p>After analysis, the right panel provides a <strong>Baseline Threshold</strong> slider. This controls the global sensitivity for detecting fibrosis. Once you set a baseline you are happy with, <strong>stick to fine area adjustments</strong> rather than changing the baseline again. Changing the baseline resets all area edits. Use <em>Reset to baseline AI estimate</em> only for a full reset.</p>
                        </div>

                        <div className="help-section">
                            <h3>Magnifying Glass &amp; Fine Adjustments</h3>
                            <p>Hover over the original image to activate the magnifying glass. Use arrow keys to zoom and adjust the threshold for just the area under observation:</p>
                            <ul className="help-bindings">
                                <li><kbd>&#9650;</kbd> <kbd>&#9660;</kbd> Zoom in / out</li>
                                <li><kbd>&#9664;</kbd> <kbd>&#9654;</kbd> Decrease / increase area threshold</li>
                                <li><kbd>Q</kbd> Inspect area and instantly compute extent (left lens) and stage classification (right lens) for the region under the magnifier. Auto-cancels when the cursor moves.</li>
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
                            <h3>Diagnose</h3>
                            <p>Classification is a <strong>separate step</strong> from analysis. Once you have refined the fibrosis mask to your satisfaction, click <strong>Diagnose</strong> to run VGG16 feature extraction and Fuzzy C-Means clustering on the refined mask. For large images, each tile is classified individually and results are averaged from the most severe patches.</p>
                            <p>After diagnosis, a radar chart shows probabilistic membership across four fibrosis stages. Use <strong>Re-diagnose</strong> after making further threshold adjustments. <strong>Analyze Patch Results</strong> highlights the five most severe tiles on the mask image with red outlines.</p>
                        </div>

                        <div className="help-section">
                            <h3>Exports</h3>
                            <p><strong>Download CSV</strong> exports the fibrosis percentage and classification scores (when available). <strong>Save Mask</strong> downloads the current fibrosis mask image (including any threshold adjustments).</p>
                        </div>

                        <div className="help-esc">Press Esc to close</div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ImageSubmission;
