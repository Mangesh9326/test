import hashlib
import os
import shutil

os.makedirs("cloud_upload", exist_ok=True)

def watermark_text(file_path, user_id):
    with open(file_path, "r") as f:
        content = f.read()

    watermark = f"\n<!--USER-ID: {hashlib.sha256(user_id.encode()).hexdigest()}-->"

    watermarked_file = "wm_" + os.path.basename(file_path)
    with open(watermarked_file, "w") as f:
        f.write(content + watermark)

    cloud_path = os.path.join("cloud_upload", watermarked_file)
    shutil.copy(watermarked_file, cloud_path)

    print(f" File watermarked and uploaded to: {cloud_path}")

if __name__ == "__main__":
    input_file = "sample2.txt"

    if not os.path.exists(input_file):
        with open(input_file, "w") as f:
            f.write("This is confidential content for cloud storage.")

    watermark_text(input_file, user_id="user123")