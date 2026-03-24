import openslide, os, glob

upload_dir = "/home/alex/nafld-uploads"
for f in sorted(glob.glob(os.path.join(upload_dir, "*.svs"))):
    size_mb = os.path.getsize(f) / (1024*1024)
    try:
        slide = openslide.OpenSlide(f)
        dims = slide.dimensions
        slide.close()
        print(f"OK: {os.path.basename(f)} ({size_mb:.0f}MB) -> {dims}")
    except Exception as e:
        print(f"FAIL: {os.path.basename(f)} ({size_mb:.0f}MB) -> {e}")
