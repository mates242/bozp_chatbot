"""
OneDrive utilities for downloading files from a public OneDrive share.
"""
import os
import requests
import json
import urllib.parse
import re
from pathlib import Path
import shutil

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
        # They may require special handling for downloading specific files
    
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
        
        # First, check if we can simply use local files
        local_data_path = "./processed_data"
        if os.path.exists(local_data_path):
            if os.path.exists(os.path.join(local_data_path, "index.faiss")) and os.path.exists(os.path.join(local_data_path, "index.pkl")):
                print(f"Local data found at {local_data_path}. Copying to temporary folder...")
                try:
                    # Copy the local files to the temp folder
                    shutil.copy2(os.path.join(local_data_path, "index.faiss"), os.path.join(dest_folder, "index.faiss"))
                    shutil.copy2(os.path.join(local_data_path, "index.pkl"), os.path.join(dest_folder, "index.pkl"))
                    print("Successfully copied local files to temporary folder.")
                    return dest_folder
                except Exception as e:
                    print(f"Error copying local files: {e}")
                    # Continue with OneDrive download as fallback
        
        # Clean/normalize the URL
        url = clean_onedrive_url(url)
        
        print(f"Attempting to access OneDrive folder: {url}")
        
        # Initialize browser session to handle authentication cookies if needed
        session = requests.Session()
        
        # For OneDrive shared folders, we need to:
        # 1. Access the main URL to get redirection and cookies
        # 2. Extract information from the response to build proper download URLs
        
        # First, get the folder page
        response = session.get(url)
        if response.status_code != 200:
            print(f"Failed to access OneDrive folder. Status code: {response.status_code}")
            return None
        
        # Extract the shared folder ID and other parameters from the URL or response
        # This is a simplified approach for 1drv.ms links
        try:
            print("Analyzing OneDrive share structure...")
            
            # Get any cookies that might be necessary for authentication
            cookies = session.cookies
            
            # Get the HTML content to extract necessary parameters
            content = response.text
            
            # For debugging
            print(f"URL after redirection: {response.url}")
            
            # Create a download base URL for files in the shared folder
            # We'll need to modify our approach based on the actual URL structure
            
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
                
            # For each file we're looking for
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
                        if filename == 'index.faiss' and file_response.headers.get('content-type') == 'application/x-gzip':
                            print(f"Detected gzipped file for {filename}, converting to raw format")
                            import gzip
                            # Read the gzipped file
                            with open(file_path, 'rb') as f_in:
                                content = f_in.read()
                            
                            # Check if it's actually gzipped
                            if content.startswith(b'\x1f\x8b'):
                                # Decompress and save back to the same file
                                try:
                                    decompressed_content = gzip.decompress(content)
                                    with open(file_path, 'wb') as f_out:
                                        f_out.write(decompressed_content)
                                    print(f"Successfully decompressed {filename}")
                                except Exception as decomp_err:
                                    print(f"Failed to decompress {filename}: {decomp_err}")
                            
                        print(f"Successfully downloaded {filename} to {file_path}")
                        downloaded_files += 1
                    else:
                        print(f"Direct download failed. Status: {file_response.status_code}. Trying alternate approaches...")
                        
                        # Approach 2: Try to find download links in the page content
                        # This would require parsing the HTML and finding the right links
                        # But for simplicity, we'll just try a few common URL patterns
                        
                        alt_urls = [
                            f"{url}:/:{filename}:/download",
                            f"{url}&download={filename}",
                            # For modern OneDrive formats
                            f"{url}&id={filename}&o=download",
                            f"{url}/download?id={filename}",
                            # Directly construct download URLs for OneDrive
                            f"https://api.onedrive.com/v1.0/shares/u!{shared_id if shared_id else ''}/items/root:/{filename}:/content"
                        ]
                        
                        for alt_url in alt_urls:
                            print(f"Trying alternate URL: {alt_url}")
                            alt_response = session.get(alt_url, stream=True, allow_redirects=True)
                            
                            if alt_response.status_code == 200 and alt_response.headers.get('content-type') != 'text/html':
                                file_path = os.path.join(dest_folder, filename)
                                with open(file_path, 'wb') as f:
                                    shutil.copyfileobj(alt_response.raw, f)
                                
                                # Check if the file is gzipped
                                if filename == 'index.faiss' and alt_response.headers.get('content-type') == 'application/x-gzip':
                                    print(f"Detected gzipped file for {filename}, converting to raw format")
                                    import gzip
                                    # Read the gzipped file
                                    with open(file_path, 'rb') as f_in:
                                        content = f_in.read()
                                    
                                    # Check if it's actually gzipped
                                    if content.startswith(b'\x1f\x8b'):
                                        # Decompress and save back to the same file
                                        try:
                                            decompressed_content = gzip.decompress(content)
                                            with open(file_path, 'wb') as f_out:
                                                f_out.write(decompressed_content)
                                            print(f"Successfully decompressed {filename}")
                                        except Exception as decomp_err:
                                            print(f"Failed to decompress {filename}: {decomp_err}")
                                
                                print(f"Successfully downloaded {filename} using alternate URL")
                                downloaded_files += 1
                                break
                            else:
                                print(f"Alternate download failed: {alt_response.status_code}")
                
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
            
            # Check if we downloaded any files
            if downloaded_files > 0:
                print(f"Downloaded {downloaded_files} files successfully")
                
                # Validate the downloaded files
                index_faiss_path = os.path.join(dest_folder, "index.faiss")
                index_pkl_path = os.path.join(dest_folder, "index.pkl")
                
                if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
                    print(f"Missing required files in {dest_folder}")
                    return None
                    
                # Check if index.faiss is valid or needs decompression
                with open(index_faiss_path, 'rb') as f:
                    header = f.read(4)
                    # Check for gzip format
                    if header.startswith(b'\x1f\x8b'):  # gzip magic number
                        print("Downloaded index.faiss is in gzip format, decompressing...")
                        import gzip
                        try:
                            with open(index_faiss_path, 'rb') as f_in:
                                decompressed_content = gzip.decompress(f_in.read())
                            with open(index_faiss_path, 'wb') as f_out:
                                f_out.write(decompressed_content)
                            print("Successfully decompressed index.faiss file")
                        except Exception as e:
                            print(f"Warning: Failed to decompress index.faiss: {e}")
                            # We'll let app.py handle this if needed
                    # Check for HTML content (indicates download failure)
                    elif header.startswith(b'\x0d\x0a<!') or header.startswith(b'<!DO') or header.startswith(b'<htm'):
                        print("ERROR: Downloaded file appears to be HTML instead of a FAISS index")
                        print("Removing invalid file and using local backup instead")
                        return None
                
                # Verify the downloaded files are not HTML files or zero-sized
                if os.path.getsize(index_faiss_path) < 1000:  # FAISS index is typically much larger
                    print(f"WARNING: index.faiss file is suspiciously small ({os.path.getsize(index_faiss_path)} bytes)")
                    return None
                
                # Read a bit of the file to check if it's valid
                try:
                    with open(index_faiss_path, 'rb') as f:
                        content_start = f.read(100)  # Read first 100 bytes
                        if b'<!DOCTYPE' in content_start or b'<html' in content_start:
                            print("ERROR: Downloaded FAISS file contains HTML - invalid format")
                            return None
                except Exception as e:
                    print(f"Error validating index.faiss: {e}")
                
                return dest_folder
            else:
                print("No files were downloaded from the OneDrive folder")
                return None
                
        except Exception as e:
            print(f"Error parsing OneDrive share: {e}")
            return None
    
    except Exception as e:
        print(f"Error accessing OneDrive folder: {e}")
        return None