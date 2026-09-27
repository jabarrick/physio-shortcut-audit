"""Extract a zip whose entries use Deflate64 (Windows Expand-Archive and stdlib zipfile cannot).
PILOT_LOG 15.21-15.22.  Usage: python unzip_any.py <zip> <dest>
Tries, in order: 7-Zip if installed; zipfile-inflate64 (binary wheels via the inflate64 package, no
compiler needed); tar.exe (libarchive).  zipfile-deflate64 is NOT used: it needs MSVC on Windows."""
import os, shutil, subprocess, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

def count():
    n = sum(f.lower().endswith(".mat") for _, _, fs in os.walk(dst) for f in fs)
    print(f".mat files under {dst}: {n}")
    return n

seven = shutil.which("7z") or next((p for p in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe")
                                    if os.path.exists(p)), None)
if seven:
    # PILOT_LOG 15.23: the SHU-MI zip asks for a password.  Check first, never prompt (a prompt would
    # hang the unattended overnight script), and never guess a password.
    info = subprocess.run([seven, "l", "-slt", "-p__none__", src], capture_output=True, text=True, errors="replace").stdout
    if "Encrypted = +" in info:
        n_enc = info.count("Encrypted = +")
        sys.exit(f"ENCRYPTED: {n_enc} entries in {src} are password-protected. Read the dataset README for access "
                 "conditions; nothing was extracted.")
    print("using 7-Zip:", seven)
    if subprocess.call([seven, "x", "-y", "-p__none__", f"-o{dst}", src]) == 0 and count():
        sys.exit(0)
try:
    try:
        import zipfile_inflate64  # noqa: F401  (patches zipfile to read Deflate64)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", "zipfile-inflate64"])
        import zipfile_inflate64  # noqa: F401
    print("using zipfile + zipfile-inflate64")
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)
    if count():
        sys.exit(0)
except Exception as e:  # noqa: BLE001
    print("zipfile-inflate64 route failed:", e)
tar = shutil.which("tar")
if tar:
    print("using tar:", tar)
    if subprocess.call([tar, "-xf", src, "-C", dst]) == 0 and count():
        sys.exit(0)
sys.exit("all extraction routes failed - install 7-Zip (winget install 7zip.7zip) and re-run")
