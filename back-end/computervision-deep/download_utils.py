import os
import requests
import hashlib
from urllib.parse import urlparse

def download_images(urls, target_dir):
    """
    Downloads images from a list of URLs to the target directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    downloaded_count = 0
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Generate a unique filename based on the URL hash
                hash_object = hashlib.md5(url.encode())
                file_ext = os.path.splitext(urlparse(url).path)[1]
                if not file_ext or file_ext.lower() not in ['.jpg', '.jpeg', '.png']:
                    file_ext = '.jpg' # Default to jpg if unknown
                
                filename = f"{hash_object.hexdigest()}{file_ext}"
                filepath = os.path.join(target_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded: {url} -> {filename}")
                downloaded_count += 1
            else:
                print(f"Failed to download (status {response.status_code}): {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    print(f"Total downloaded to {target_dir}: {downloaded_count}")

if __name__ == "__main__":
    # Example usage (will be replaced by actual calls)
    pass
