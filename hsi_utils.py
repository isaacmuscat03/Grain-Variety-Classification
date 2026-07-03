from pathlib import Path
import pandas as pd
import subprocess, os
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, disk, binary_erosion, disk
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.ndimage import median_filter
from skimage import exposure

def load_data():
    # Mounting dataset from D: to WSL 
    WINDOWS_DRIVE = "E:"   
    WINDOWS_PATH = r"E:\HSI_Dataset_2\Elements\data"
    LINK_NAME     = "data_external"  

    # Mount drive if missing
    mnt_path = Path(f"/mnt/{WINDOWS_DRIVE[0].lower()}")

    print(f" Mounting {WINDOWS_DRIVE} into {mnt_path} ...")
    subprocess.run(["sudo", "mkdir", "-p", str(mnt_path)], check=True)
    res = subprocess.run(["sudo", "mount", "-t", "drvfs", WINDOWS_DRIVE, str(mnt_path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to mount {WINDOWS_DRIVE}: {res.stderr}")

    # Verify dataset path 
    dataset_path = Path(str(WINDOWS_PATH).replace("\\", "/").replace(":", "").replace("E", "/mnt/e", 1))
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}. Let's check what's under /mnt/e:")
        os.system("ls -la /mnt/d")
        raise FileNotFoundError("Fix dataset path above and rerun.")
    print(f"[OK] Found dataset: {dataset_path}")

    # Create symlink inside project
    proj_root = Path.cwd()
    link_path = proj_root / LINK_NAME
    if link_path.exists() or link_path.is_symlink():
        print(f" Removing old link {link_path}")
        link_path.unlink()
    link_path.symlink_to(dataset_path, target_is_directory=True)
    print(f"[OK] Linked {link_path} -> {dataset_path}")

    # Show a few sample files for confirmation
    import itertools
    exts = {".hdf5", ".h5", ".hdr", ".tif", ".tiff"}
    found = list(itertools.islice((p for p in link_path.rglob("*") if p.suffix.lower() in exts), 10))
    if found:
        print("Sample files:")
        for f in found: print("  ", f.relative_to(link_path))
    else:
        print("No .hdf5/.h5/.hdr/.tif files found yet — check deeper folders.")

    ROOT = Path("data_external")  
    HR_ROOT = ROOT / "raw" / "FX10" 

    OUTDIR = HR_ROOT.parent.parent / "processed" / "quickrun"  
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("ROOT:    ", ROOT.resolve())
    print("Basepath:", HR_ROOT.resolve())
    print("OUTDIR:  ", OUTDIR.resolve())

    ## Look for all files in the folder that end with .hdf5 
    df = pd.DataFrame({"filepath_FX10": list(Path(f"{HR_ROOT}").rglob("**/*.hdf5"))})
    ## Give the sample name 
    df['sample_name'] = df.filepath_FX10.apply(lambda x : x.stem)
    df

    # find all .hdf5 files
    all_files = list(HR_ROOT.rglob("*.hdf5"))
    print("Total .hdf5 files found:", len(all_files))

    df_files = pd.DataFrame({"filepath_FX10": all_files})
    df_files["sample_name"] = df_files["filepath_FX10"].apply(lambda p: p.stem)


    def resolve_valid_hdf5(path: Path) -> Path | None:
        """
        Returns:
            - Path to a valid file or None if it fails
        """
        path = Path(path)

        try:
            with h5py.File(path, "r"):
                return path
        except Exception:
            return None

    df_files["resolved_path"] = df_files["filepath_FX10"].apply(resolve_valid_hdf5)

    df_valid = df_files[df_files["resolved_path"].notna()].copy()
    df_invalid = df_files[df_files["resolved_path"].isna()].copy()

    print("Valid HDF5 files:", len(df_valid))
    print("Invalid/unreadable files:", len(df_invalid))

    if not df_invalid.empty:
        print("\nUnreadable sample_names:")
        print(df_invalid["sample_name"].tolist())

    # --- Create image-level train/valid/test split ---

    df_valid = df_valid.copy()

    df_valid["variety"] = df_valid["sample_name"].str.split("_").str[0]
    df_valid["batch"]   = df_valid["sample_name"].str.split("_").str[-1]

    df_valid["size"] = df_valid["batch"].str[0]
    df_valid["rep"]  = df_valid["batch"].str[1:].astype(int)

    varieties = ["barley", "corn", "flax"]

    df_valid = df_valid[df_valid["variety"].isin(varieties)].copy()

    def assign_split(group):
        group = group.sort_values("rep").copy()

        if len(group) != 5:
            print("Warning: expected 5 images, got", len(group), "for", group[["variety", "size"]].iloc[0].to_dict())

        group["split"] = "unused"
        group.iloc[:3, group.columns.get_loc("split")] = "train"
        group.iloc[3:4, group.columns.get_loc("split")] = "valid"
        group.iloc[4:5, group.columns.get_loc("split")] = "test"

        return group

    files_split = (
        df_valid
        .groupby(["variety", "size"], group_keys=False)
        .apply(assign_split)
        .reset_index(drop=True)
    )

    train_files = files_split[files_split["split"] == "train"].copy()
    valid_files = files_split[files_split["split"] == "valid"].copy()
    test_files  = files_split[files_split["split"] == "test"].copy()

    train_files.loc[:, 'variety'] = train_files['sample_name'].str.split("_").str[0]
    train_files.loc[:, 'batch']   = train_files['sample_name'].str.split("_").str[-1]

    train_files.loc[:, 'size'] = train_files['batch'].str[0]          
    train_files.loc[:, 'rep']  = train_files['batch'].str[1:].astype(int)   

    return train_files, valid_files, test_files, files_split

def load_cube(path, verbose=False):
    with h5py.File(path, "r") as f:
        hcube = np.array(f["hypercube"][:,:,:]) / 10000
        darkref = np.array(f["dark_reference"]) / 10000
        whiteref = np.array(f["white_reference"]) / 10000
        wlens = f["hypercube"].attrs["wavelength_nm"]

        hcube = np.swapaxes(hcube, -1, 0).astype("float32")
        hcube = np.fliplr(hcube)

    return hcube, wlens, darkref, whiteref
    
def visualize_hcube_FX10(path_FX10):
    # Load hyperspectral cube
    hcube, wlens, darkref, whiteref = load_cube(path_FX10)
    wlens = wlens.astype(int)

    # Show 25 random bands
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

    channels = random.sample(range(len(wlens)), 25)
    channels = np.sort(channels)

    for i, channel in enumerate(channels):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(hcube[:, :, channel], cmap='viridis')
        axs[row, col].set_title(f"band {channel}, λ={wlens[channel]} nm")
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def make_rgb(hcube, band_ids=(60, 108, 163)):
    H, W, B = hcube.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    for i, b in enumerate(band_ids):
        v = hcube[:, :, b]
        lo, hi = np.percentile(v, (2, 98))  
        rgb[:, :, i] = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)

    return rgb

def get_mask_params(size, true_label=None):
    # Default fallback
    params = dict(
        sigma=250,
        med_size=3,
        closing_radius=0,
        opening_radius=0,
        erosion=0,
        bin_iterations=2,
        min_size=1,
        area_threshold=0
    )

    # Large grains
    if size == "l":
        params.update(
            sigma=180,
            med_size=3,
            closing_radius=5,
            opening_radius=0,
            erosion=0,
            bin_iterations=4,
            min_size=1,
            area_threshold=30
        )

        if true_label == "corn":
            params.update(
                sigma=170,
                closing_radius=7,
                bin_iterations=6,
                area_threshold=1
            )

        elif true_label == "flax":
            params.update(
                sigma=250,
                closing_radius=7,
                opening_radius=2,
                bin_iterations=3,
                min_size=40,
                area_threshold=140,
                use_clahe=True,
                clahe_clip=0.02
            )

    # Medium grains
    elif size == "m":
        params.update(
            sigma=250,
            med_size=3,
            closing_radius=1,
            opening_radius=2,
            erosion=0,
            bin_iterations=1,
            min_size=1,
            area_threshold=0
        )

        if true_label == "flax":
            params.update(
                closing_radius=7,
                opening_radius=2,
                bin_iterations=3,
                min_size=40,
                area_threshold=120,
                use_clahe=True,
                clahe_clip=0.02
            )

    # Small grains
    elif size == "s":
        if true_label == "flax":
            params.update(
                sigma=250,
                med_size=3,
                closing_radius=3,
                opening_radius=0,
                erosion=0,
                bin_iterations=3,
                min_size=1,
                area_threshold=0
            )

    return params


def build_foreground_mask_auto(hcube, size, true_label=None):
    mask_params = get_mask_params(size=size, true_label=true_label)

    return build_foreground_mask(
        hcube,
        **mask_params
    )

def build_foreground_mask(
    hcube,
    sigma=60,
    med_size=3,
    closing_radius=3,
    opening_radius=1,
    erosion=1,
    bin_iterations=2,
    min_size=1,
    area_threshold=50,
    use_clahe=False,
    clahe_clip=0.02
):

    mean_img = hcube.mean(axis=2)

    # Robust normalize
    lo, hi = np.percentile(mean_img, (1, 99))
    mean_norm = np.clip((mean_img - lo) / (hi - lo + 1e-6), 0, 1)

    # Estimate smooth illumination/background field
    illumination = gaussian_filter(mean_norm, sigma=sigma)

    # Correct left-right gradient
    mean_corr = mean_norm - illumination
    
    # Normalize corrected image again
    lo2, hi2 = np.percentile(mean_corr, (1, 99))
    mean_corr = np.clip((mean_corr - lo2) / (hi2 - lo2 + 1e-6), 0, 1)

    mean_corr = median_filter(mean_corr, size=med_size)

    # adapting to darker / low-contrast grains like flax
    if use_clahe:
        mean_corr = exposure.equalize_adapthist(
            mean_corr,
            clip_limit=clahe_clip
        )

    # Global Otsu threshold
    thresh = threshold_otsu(mean_corr)
    binary = mean_corr > thresh

    binary = closing(binary, disk(closing_radius))
    binary = opening(binary, disk(opening_radius))

    binary = binary_erosion(binary, disk(erosion))
    binary = binary_dilation(binary, iterations=bin_iterations)

    binary = remove_small_objects(binary, min_size=min_size)
    if area_threshold > 0:
        binary = remove_small_holes(binary, area_threshold=area_threshold)
        
    return binary