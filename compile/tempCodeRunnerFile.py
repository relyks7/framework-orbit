import os
import shutil
import subprocess

# --- Use xcrun to locate tools dynamically ---
# We use a list here so we can prepend it to the subprocess arguments later
METAL_CMD = ["xcrun", "-sdk", "macosx", "metal"]
METALLIB_CMD = ["xcrun", "-sdk", "macosx", "metallib"]

SRC_DIR = "metal"
OUT_DIR = "kernels"

# Clean output
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

# Compile all .metal files
for root, dirs, files in os.walk(SRC_DIR):
    for fname in files:
        if not fname.endswith(".metal"):
            continue

        src_path = os.path.join(root, fname)
        name = os.path.splitext(fname)[0]
        air_path = os.path.join(OUT_DIR, f"{name}.air")
        lib_path = os.path.join(OUT_DIR, f"{name}.metallib")

        # Compile → .air
        # We unwrap the list METAL_CMD into the subprocess call
        run(METAL_CMD + ["-c", src_path, "-o", air_path]) 

        # Link → .metallib
        run(METALLIB_CMD + [air_path, "-o", lib_path])

print("[+] Compiled kernels.")