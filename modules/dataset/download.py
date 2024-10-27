"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

"""

import argparse
import os
import gdown
import subprocess
import zipfile
import tarfile

def download_megadepth_1500(download_dir):
    # Google Drive files (test indices)
    files = {
        'test_images': 'https://drive.google.com/uc?id=12yKniNWebDHRTCwhBNJmxYMPgqYX3Nhv',
    }

    os.makedirs(download_dir, exist_ok=True)

    # Download Google Drive files with gdown
    for file_name, url in files.items():
        output_path = os.path.join(download_dir, f'{file_name}.tar')
        gdown.download(url, output_path, quiet=False)

        # Extract the tar.gz file
        if tarfile.is_tarfile(output_path):
            print(f"Extracting {output_path}...")
            with tarfile.open(output_path, 'r:tar') as tar:
                tar.extractall(path=download_dir)

            os.remove(output_path)

def download_scannet_1500(download_dir):
    files = {
        'test_images': 'https://drive.google.com/uc?id=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd',
        'gt_poses': 'https://github.com/zju3dv/LoFTR/raw/refs/heads/master/assets/scannet_test_1500/test.npz',
    }
    
    os.makedirs(download_dir, exist_ok=True)

    # Download Google Drive files with gdown
    for file_name, url in files.items():
        if 'drive.google.com' in url:
            output_path = os.path.join(download_dir, f'{file_name}.tar')
            gdown.download(url, output_path, quiet=False)

            # Extract the tar.gz file
            if tarfile.is_tarfile(output_path):
                print(f"Extracting {output_path}...")
                with tarfile.open(output_path, 'r:tar') as tar:
                    tar.extractall(path=download_dir)

                os.remove(output_path)
        elif 'github.com' in url:
            fname = url.split('/')[-1]
            output_path = os.path.join(download_dir, fname)
            subprocess.run(['wget', '-c', url, '-O', output_path])

def download_megadepth(download_dir):

    response = input("Warning: MegaDepth requires about 500 GB of free disk space. Continue? [y/n]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Exiting the program.")
        exit(0)

    os.makedirs(download_dir, exist_ok=True)

    # Google Drive files (training indices and test indices)
    files = {
        'train_test_indices': 'https://drive.google.com/uc?id=1YMAAqCQLmwMLqAkuRIJLDZ4dlsQiOiNA',
    }

    # Download Google Drive files with gdown
    for file_name, url in files.items():
        output_path = os.path.join(download_dir, f'{file_name}.tar')
        gdown.download(url, output_path, quiet=False)

        # Extract the tar.gz file
        if tarfile.is_tarfile(output_path):
            print(f"Extracting {output_path}...")
            with tarfile.open(output_path, 'r:tar') as tar:
                tar.extractall(path=download_dir)

    # Training images via wget
    training_data_url = 'https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz'
    training_data_path = os.path.join(download_dir, 'Megadepth_v1.tar.gz')
    subprocess.run(['wget', '-c', training_data_url, '-O', training_data_path])

    # Extract the training data
    if tarfile.is_tarfile(training_data_path):
        print(f"Extracting {training_data_path}...")
        with tarfile.open(training_data_path, 'r:gz') as tar:
            tar.extractall(path=download_dir)

        os.remove(training_data_path)

def main():
    parser = argparse.ArgumentParser(description="Datasets Downloader Tool. All links are used from LoFTR, HPatches and MegaDepth original papers.")
    
    parser.add_argument('--megadepth', action='store_true', help="Download MegaDepth dataset")
    parser.add_argument('--megadepth-1500', action='store_true', help="Download MegaDepth-1500 test set")
    parser.add_argument('--scannet-1500', action='store_true', help="Download ScanNet-1500 test set")
    parser.add_argument('--hpatches', action='store_true', help="Download HPatches dataset (not implemented)")
    parser.add_argument('--download_dir', required=True, type=str, help="Directory to download datasets")
    
    args = parser.parse_args()
    
    if args.megadepth:
        print(f"Downloading MegaDepth dataset to [{args.download_dir}]")
        download_megadepth(args.download_dir + '/MegaDepth')
    elif args.megadepth_1500:
        print(f"Downloading MegaDepth-1500 dataset to [{args.download_dir}]")
        download_megadepth_1500(args.download_dir + '/Mega1500')
    elif args.scannet_1500:
        print(f"Downloading ScanNet dataset to [{args.download_dir}]")
        download_scannet_1500(args.download_dir + '/ScanNet1500')
    else:
        raise RuntimeError("Dataset not implemented for download.")

if __name__ == '__main__':
    main()
