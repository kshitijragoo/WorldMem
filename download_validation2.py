import os
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

def download_specific_folder(repo_id, folder_name, local_dir_root):
    """
    Downloads a specific folder from a Hugging Face Hub repository.
    """
    # Create the root directory if it doesn't exist
    # e.g., 'data/minecraft'
    os.makedirs(local_dir_root, exist_ok=True)
    
    # Get a list of all files in the repository
    all_files = list_repo_files(repo_id, repo_type='dataset')

    # Filter for files that are in the desired folder (e.g., 'validation/')
    target_files = [f for f in all_files if f.startswith(folder_name)]

    if not target_files:
        print(f"No files found in folder '{folder_name}' in repository '{repo_id}'.")
        return

    print(f"Found {len(target_files)} files in '{folder_name}'. Starting download...")

    # Download each file into the correct local subdirectory
    for filename in tqdm(target_files, desc=f"Downloading {folder_name}"):
        destination_path = os.path.join(local_dir_root, filename)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type='dataset',
            local_dir=local_dir_root,
            local_dir_use_symlinks=False
        )
    
    print(f"✅ Download complete. Files are in '{os.path.join(local_dir_root, folder_name)}'")

# --- Configuration ---
DATASET_REPO_ID = "zeqixiao/worldmem_minecraft_dataset"
FOLDER_TO_DOWNLOAD = "validation/"
LOCAL_PROJECT_ROOT = "data/minecraft"

# --- Run the script ---
if __name__ == "__main__":
    # --- NEW LINES START ---
    # Create the empty training directory structure
    training_dir_path = os.path.join(LOCAL_PROJECT_ROOT, 'training')
    os.makedirs(training_dir_path, exist_ok=True)
    print(f"Ensured empty directory exists at: {training_dir_path}")
    # --- NEW LINES END ---

    # Call the function to download the validation data
    download_specific_folder(DATASET_REPO_ID, FOLDER_TO_DOWNLOAD, LOCAL_PROJECT_ROOT)