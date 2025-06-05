"""
OneDrive utilities for downloading files from a public OneDrive share.
"""
import os
import requests
from pathlib import Path
import shutil
import base64

def clean_onedrive_url(url):
    """Clean and normalize OneDrive URL to ensure we can work with it."""
    # Replace "1drv.ms" short links with the full onedrive.live.com URL if needed
    if "1drv.ms" in url:
        # For 1drv.ms links, we need to follow the redirect to get the actual URL
        try:
            response = requests.head(url, allow_redirects=True)
            url = response.url
            print(f"Resolved OneDrive URL: {url}")
        except Exception as e:
            print(f"Error resolving OneDrive short URL: {e}")
            return url
    
    # For specific OneDrive formats with '/f/c/' pattern (newer OneDrive links)
    if '/f/c/' in url:
        print(f"Detected modern OneDrive link format: {url}")
        # These URLs typically need to be accessed directly
    
    return url

def download_onedrive_folder(url, dest_folder):
    """
    Downloads files from a public OneDrive folder to a local destination folder.
    Specifically designed to handle vectorstore files (index.faiss, index.pkl).
    
    Args:
        url: Public OneDrive URL to the folder
        dest_folder: Local destination folder path
    
    Returns:
        Path to the destination folder if successful, None otherwise
    """
    try:
        # Create destination folder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)
        
        # Always download fresh copy from OneDrive as per requirements
        print("Attempting to download data from OneDrive...")
        
        # Clean/normalize the URL
        url = clean_onedrive_url(url)
        
        print(f"Attempting to access OneDrive folder: {url}")
        
        # Try downloading using a predefined direct download link first
        # These are verified direct download URLs that work with the new OneDrive link
        direct_download_urls = {
            "index.faiss": "https://onedrive.live.com/download?resid=6C73ABCE92E1E313%21946&authkey=!ABzVSueQFLDmdWGl&e=0MZgTF",
            "index.pkl": "https://onedrive.live.com/download?resid=6C73ABCE92E1E313%21947&authkey=!ABzVSueQFLDmdWGl&e=QsXLzR"
        }
        
        # Try direct download first - this is the most reliable method
        print("Attempting direct download using pre-configured URLs...")
        download_success = download_using_predefined_urls(direct_download_urls, dest_folder)
        if download_success:
            print("Successfully downloaded files using direct download URLs")
            return dest_folder
        
        # Initialize browser session to handle authentication cookies if needed
        session = requests.Session()
        
        # First, get the folder page
        response = session.get(url)
        if response.status_code != 200:
            print(f"Failed to access OneDrive folder. Status code: {response.status_code}")
            return None
        
        # Try downloading using alternative methods
        try:
            print("Analyzing OneDrive share structure...")
            
            # Get any cookies that might be necessary for authentication
            cookies = session.cookies
            
            # Get the HTML content to extract necessary parameters
            content = response.text
            
            # For debugging
            print(f"URL after redirection: {response.url}")
            
            # Extract the shared folder ID from the URL
            shared_id = None
            if "?" in response.url:
                base_url = response.url.split("?")[0]
                if "/f/" in base_url:
                    parts = base_url.split("/f/")
                    if len(parts) > 1 and "/" in parts[1]:
                        shared_id = parts[1].split("/")[0]
            
            if shared_id:
                print(f"Detected shared folder ID: {shared_id}")
            else:
                print("Could not determine shared folder ID from URL")

            # Try downloading from direct file links
            use_direct_links = True
            if use_direct_links:
                # Hardcoded direct download links to the files (base64 encoded to prevent truncation)
                encoded_links = {
                    "index.faiss": "aHR0cHM6Ly9vbmVkcml2ZS5saXZlLmNvbS9kb3dubG9hZD9jb2RlPXZ3YjVwSkxDS1RmQ2V6Z1MweUVrWFEwamNpYlpuRWhhaW94V29NVFhwYTFOSUEmUmVzb3VyY2VLZXk9WFZYc0lWSzMxbmRXYnc=",
                    "index.pkl": "aHR0cHM6Ly9vbmVkcml2ZS5saXZlLmNvbS9kb3dubG9hZD9jb2RlPXZ3YjVwSkxDS1RmQ2V6Z1MweUVrWFEwamNpYlpuRWhhaW94V29NVFhwYTFOSUEmUmVzb3VyY2VLZXk9TzNDdURwUXJDRUxzcUhk"
                }
                
                downloaded_files = 0
                for filename, encoded_url in encoded_links.items():
                    try:
                        direct_url = base64.b64decode(encoded_url).decode('utf-8')
                        print(f"Attempting direct download of {filename} using direct link")
                        file_response = session.get(direct_url, stream=True, allow_redirects=True)
                        
                        if file_response.status_code == 200:
                            file_path = os.path.join(dest_folder, filename)
                            with open(file_path, 'wb') as f:
                                shutil.copyfileobj(file_response.raw, f)
                            
                            # Validate the downloaded file
                            is_valid = validate_downloaded_file(file_path, filename)
                            if is_valid:
                                print(f"Successfully downloaded {filename} to {file_path}")
                                downloaded_files += 1
                            else:
                                print(f"Downloaded {filename} appears invalid")
                    except Exception as e:
                        print(f"Error downloading {filename} from direct link: {e}")
            
            # If direct links failed, try standard OneDrive methods
            if not downloaded_files >= 2:
                downloaded_files = try_standard_onedrive_download(url, dest_folder, session, shared_id)
            
            # Check if we downloaded any files
            if downloaded_files >= 2:  # We need both index.faiss and index.pkl
                print(f"Downloaded {downloaded_files} files successfully")
                
                # Final validation of the downloaded files
                if not is_vectorstore_valid(dest_folder):
                    print("Downloaded files failed validation")
                    return use_local_backup(dest_folder)
                
                return dest_folder
            else:
                print("Not all required files were downloaded from the OneDrive folder")
                return use_local_backup(dest_folder)
                
        except Exception as e:
            print(f"Error parsing OneDrive share: {e}")
            return use_local_backup(dest_folder)
    
    except Exception as e:
        print(f"Error accessing OneDrive folder: {e}")
        return use_local_backup(dest_folder)

def try_standard_onedrive_download(url, dest_folder, session, shared_id):
    """Try downloading using standard OneDrive methods"""
    downloaded_files = 0
    for filename in ["index.faiss", "index.pkl"]:
        try:
            # Approach 1: Try direct download from the folder
            # For modern OneDrive links with /f/c/ pattern
            if '/f/c/' in url:
                # For newer OneDrive links with format /f/c/, try these variations
                if url.endswith('/'):
                    direct_url = f"{url}{filename}"
                else:
                    direct_url = f"{url}/{filename}"
                # Add download parameter if needed
                if '?' in direct_url:
                    direct_url += f"&download=1"
                else:
                    direct_url += f"?download=1"
            else:
                direct_url = f"{url}/{filename}?download=1"
            print(f"Attempting to download {filename} from: {direct_url}")
            
            file_response = session.get(direct_url, stream=True, allow_redirects=True)
            
            if file_response.status_code == 200 and file_response.headers.get('content-type') != 'text/html':
                # Save the file
                file_path = os.path.join(dest_folder, filename)
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(file_response.raw, f)
                    
                # Check if the file is gzipped (common for OneDrive downloads)
                if filename == 'index.faiss':
                    decompress_if_needed(file_path)
                
                if validate_downloaded_file(file_path, filename):
                    print(f"Successfully downloaded {filename} to {file_path}")
                    downloaded_files += 1
                else:
                    print(f"Downloaded {filename} appears to be invalid")
            else:
                print(f"Direct download failed. Status: {file_response.status_code}. Trying alternate approaches...")
                
                # Approach 2: Try to find download links in the page content
                alt_urls = [
                    f"{url}:/:{filename}:/download",
                    f"{url}&download={filename}",
                    f"{url}/download?id={filename}",
                    f"https://api.onedrive.com/v1.0/shares/u!{shared_id if shared_id else ''}/items/root:/{filename}:/content"
                ]
                
                for alt_url in alt_urls:
                    print(f"Trying alternate URL: {alt_url}")
                    alt_response = session.get(alt_url, stream=True, allow_redirects=True)
                    
                    if alt_response.status_code == 200 and alt_response.headers.get('content-type') != 'text/html':
                        file_path = os.path.join(dest_folder, filename)
                        with open(file_path, 'wb') as f:
                            shutil.copyfileobj(alt_response.raw, f)
                        
                        # Check if gzipped
                        if filename == 'index.faiss':
                            decompress_if_needed(file_path)
                        
                        if validate_downloaded_file(file_path, filename):
                            print(f"Successfully downloaded {filename} using alternate URL")
                            downloaded_files += 1
                            break
                        else:
                            print(f"Downloaded {filename} appears to be invalid")
                    else:
                        print(f"Alternate download failed: {alt_response.status_code}")
        
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    return downloaded_files

def decompress_if_needed(file_path):
    """Decompress the file if it's gzipped"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header.startswith(b'\x1f\x8b'):  # gzip magic number
                print(f"Detected gzipped file {file_path}, decompressing...")
                import gzip
                try:
                    with open(file_path, 'rb') as f_in:
                        decompressed_content = gzip.decompress(f_in.read())
                    with open(file_path, 'wb') as f_out:
                        f_out.write(decompressed_content)
                    print("Successfully decompressed file")
                    return True
                except Exception as e:
                    print(f"Failed to decompress file: {e}")
                    return False
    except Exception as e:
        print(f"Error checking for compression: {e}")
        return False
    return True  # Not compressed or already handled

def validate_downloaded_file(file_path, filename):
    """Validate that the downloaded file is of correct format"""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:
            print(f"File {filename} is too small or doesn't exist")
            return False
            
        # For index.faiss, check if it's not HTML
        if filename == "index.faiss":
            with open(file_path, 'rb') as f:
                header = f.read(100)
                if (b'<!DOCTYPE' in header or b'<html' in header or 
                    b'<HTML' in header or b'HTTP/' in header or 
                    b'\x0d\x0a<!' in header):
                    print(f"File {filename} contains HTML - invalid format")
                    return False
                    
        # For index.pkl, minimal validation
        if filename == "index.pkl" and os.path.getsize(file_path) < 1000:
            print(f"File {filename} is suspiciously small")
            return False
            
        return True
    except Exception as e:
        print(f"Error validating {filename}: {e}")
        return False

def is_vectorstore_valid(folder_path):
    """Validate that both files required for the vectorstore exist and are valid"""
    index_faiss_path = os.path.join(folder_path, "index.faiss")
    index_pkl_path = os.path.join(folder_path, "index.pkl")
    
    if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
        print(f"Missing required files in {folder_path}")
        return False
        
    if os.path.getsize(index_faiss_path) < 1000 or os.path.getsize(index_pkl_path) < 100:
        print(f"Files in {folder_path} are suspiciously small")
        return False
        
    # Check if index.faiss is valid
    try:
        with open(index_faiss_path, 'rb') as f:
            header = f.read(100)
            if (b'<!DOCTYPE' in header or b'<html' in header or 
                b'<HTML' in header or b'HTTP/' in header or 
                b'\x0d\x0a<!' in header):
                print(f"index.faiss contains HTML - invalid format")
                return False
    except Exception as e:
        print(f"Error validating index.faiss: {e}")
        return False
        
    return True

def use_local_backup(dest_folder):
    """Try to use local backup files if available"""
    # Check common backup locations
    backup_locations = [
        "./stiahnute_zakony",
        "./backup",
        "./data",
        os.path.join(os.path.dirname(os.path.dirname(dest_folder)), "stiahnute_zakony"),
        os.path.join(os.path.dirname(os.path.dirname(dest_folder)), "backup"),
        os.path.join(os.path.dirname(os.path.dirname(dest_folder)), "data"),
    ]
    
    for backup_dir in backup_locations:
        local_index_faiss = os.path.join(backup_dir, "index.faiss")
        local_index_pkl = os.path.join(backup_dir, "index.pkl")
        
        if os.path.exists(local_index_faiss) and os.path.exists(local_index_pkl):
            print(f"Found backup files in {backup_dir}")
            
            # Check if they're valid
            if validate_downloaded_file(local_index_faiss, "index.faiss") and validate_downloaded_file(local_index_pkl, "index.pkl"):
                try:
                    # Copy the backup files
                    shutil.copy2(local_index_faiss, os.path.join(dest_folder, "index.faiss"))
                    shutil.copy2(local_index_pkl, os.path.join(dest_folder, "index.pkl"))
                    print(f"Successfully copied backup files from {backup_dir} to {dest_folder}")
                    return dest_folder
                except Exception as e:
                    print(f"Error copying backup files: {e}")
    
    print("No valid backup files found")
    return None

def download_using_predefined_urls(urls, dest_folder):
    """Download using predefined direct download URLs"""
    success_count = 0
    
    for filename, url in urls.items():
        try:
            print(f"Downloading {filename} from predefined URL")
            
            # Use stream=True to handle large files
            with requests.get(url, stream=True, timeout=30) as response:
                if response.status_code == 200:
                    file_path = os.path.join(dest_folder, filename)
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Validate the downloaded file
                    if validate_downloaded_file(file_path, filename):
                        print(f"Successfully downloaded {filename}")
                        success_count += 1
                    else:
                        print(f"Downloaded file {filename} is invalid")
                else:
                    print(f"Failed to download {filename}: HTTP status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    # Return True if all files were downloaded successfully
    return success_count == len(urls)

def download_using_alternative_urls(base_url, dest_folder):
    """Try downloading using alternative URL construction methods"""
    filenames = ["index.faiss", "index.pkl"]
    success_count = 0
    
    # Some alternative URL patterns that might work
    url_patterns = [
        "{base_url}/{filename}?download=1",
        "{base_url}/download/{filename}",
        "{base_url}/{filename}?dl=1",
        "{base_url}:/:{filename}:/content"
    ]
    
    for filename in filenames:
        for pattern in url_patterns:
            try:
                url = pattern.format(base_url=base_url, filename=filename)
                print(f"Trying to download {filename} using URL: {url}")
                
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    # Check if response appears to be HTML instead of binary data
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type.lower():
                        print(f"Received HTML instead of file data for {filename}")
                        continue
                        
                    file_path = os.path.join(dest_folder, filename)
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Check if we got HTML content
                    with open(file_path, 'rb') as f:
                        first_bytes = f.read(100)
                        if b'<!DOCTYPE' in first_bytes or b'<html' in first_bytes:
                            print(f"Downloaded content for {filename} appears to be HTML, not a valid file")
                            continue
                    
                    if filename == "index.faiss":
                        decompress_if_needed(file_path)
                        
                    if validate_downloaded_file(file_path, filename):
                        print(f"Successfully downloaded {filename} using URL pattern: {pattern}")
                        success_count += 1
                        break
            except Exception as e:
                print(f"Error downloading {filename} using URL pattern {pattern}: {e}")
    
    return success_count == len(filenames)

def download_using_curl(dest_folder, direct_urls):
    """Try downloading using curl command (system dependent)"""
    import subprocess
    
    success_count = 0
    filenames = ["index.faiss", "index.pkl"]
    
    for filename in filenames:
        try:
            file_path = os.path.join(dest_folder, filename)
            url = direct_urls.get(filename)
            
            if not url:
                print(f"No direct URL available for {filename}")
                continue
                
            print(f"Attempting to download {filename} using curl")
            
            # Use curl to download the file
            result = subprocess.run(
                ["curl", "-L", "-o", file_path, url],
                capture_output=True
            )
            
            if result.returncode == 0 and os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                if filename == "index.faiss":
                    decompress_if_needed(file_path)
                
                if validate_downloaded_file(file_path, filename):
                    print(f"Successfully downloaded {filename} using curl")
                    success_count += 1
                else:
                    print(f"Downloaded {filename} appears invalid")
            else:
                print(f"Curl download of {filename} failed: {result.stderr.decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"Error using curl to download {filename}: {e}")
    
    return success_count == len(filenames)

def create_minimal_vectorstore(dest_folder):
    """Create a minimal vectorstore from scratch when downloads fail"""
    print("Creating minimal vectorstore from scratch...")
    
    try:
        # This function requires OpenAI API key which should be set in the environment or passed explicitly
        # Import necessary dependencies here to avoid circular imports
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        import os
        
        # Check if OpenAI API key is available in the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not available in environment. Cannot create minimal vectorstore.")
            return None
            
        # Create minimal data to build a vectorstore
        texts = [
            "Toto je záložná databáza vytvorená pre fungovanie aplikácie.",
            "Obsahuje len minimálne množstvo údajov pre spustenie systému.",
            "Občiansky zákonník upravuje majetkové vzťahy fyzických a právnických osôb.",
            "Trestný zákon definuje, čo je trestným činom a aké sú tresty.",
            "Ústava Slovenskej republiky je základným zákonom štátu."
        ]
        
        metadatas = [
            {"source": "fallback/system.txt"},
            {"source": "fallback/system.txt"},
            {"source": "fallback/obciansky_zakonnik.pdf"},
            {"source": "fallback/trestny_zakon.pdf"},
            {"source": "fallback/ustava.pdf"}
        ]
        
        # Create embeddings and vectorstore
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        # Save the vectorstore
        vectorstore.save_local(dest_folder)
        
        # Verify that the files were created correctly
        if os.path.exists(os.path.join(dest_folder, "index.faiss")) and \
           os.path.exists(os.path.join(dest_folder, "index.pkl")):
            print("Successfully created minimal vectorstore")
            return dest_folder
        else:
            print("Failed to create minimal vectorstore")
            return None
            
    except Exception as e:
        print(f"Error creating minimal vectorstore: {e}")
        return None