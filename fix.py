import os
path = r'NAFLD/javascript/nafld-app/src/ImageSubmission.js'
with open(path, 'r', encoding='utf-8') as f: content = f.read()

content = content.replace(\"{isLoadingExcluded ? '…' : showExcluded ? '? Excluded' : '? Excluded'}\", \"{isLoadingExcluded ? '…' : showExcluded ? '? what was excluded?' : '? what was excluded?'}\")

old_btn = \"\"\"{displayedMaskSrc && (
                            <button className=\"download-mask-btn\" onClick={handleDownloadMask} title=\"Download fibrosis mask\">
                                ? Save Mask
                            </button>
                        )}\"\"\"

new_btn = \"\"\"{displayedMaskSrc && (
                            !analysisResult ? (
                                <div className=\"download-mask-btn\" style={{ color: '#a855f7', borderColor: '#a855f7', pointerEvents: 'none', cursor: 'default', background: 'transparent' }}>
                                    preview only
                                </div>
                            ) : (
                                <button className=\"download-mask-btn\" onClick={handleDownloadMask} title=\"Download fibrosis mask\">
                                    ? Save Mask
                                </button>
                            )
                        )}\"\"\"

if old_btn in content:
    content = content.replace(old_btn, new_btn)
else:
    print('old_btn not found')

with open(path, 'w', encoding='utf-8') as f: f.write(content)
print('Done!')
