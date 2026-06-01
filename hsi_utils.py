from pathlib import Path
import pandas as pd
import subprocess, os
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def load_data():
    # Mounting dataset from D: to WSL 
    WINDOWS_DRIVE = "E:"   
    WINDOWS_PATH = r"E:\HSI_Dataset_2\Elements\data"
    LINK_NAME     = "data_external"  

    # Mount drive if missing
    mnt_path = Path(f"/mnt/{WINDOWS_DRIVE[0].lower()}")
    #if not mnt_path.exists():
    print(f" Mounting {WINDOWS_DRIVE} into {mnt_path} ...")
    subprocess.run(["sudo", "mkdir", "-p", str(mnt_path)], check=True)
    res = subprocess.run(["sudo", "mount", "-t", "drvfs", WINDOWS_DRIVE, str(mnt_path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to mount {WINDOWS_DRIVE}: {res.stderr}")
    #else:
    #    print(f"[OK] {mnt_path} already exists")

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

    with h5py.File(path, 'r') as f:
        f = h5py.File(path)

        if verbose:
            print("What is inside an h5 file?\n")
            for key in f.keys(): print(f"key:'{key}', \nin which there is '{f[key]}' \
            \nwhich we load in a numpy array of shape: {np.array(f[key]).shape}\n");
        
            print(f"Selected to load cube from file:\n{path}")

        hcube = np.array(f['hypercube'][:,:,:])/10000  
        darkref = np.array(f['dark_reference'])/10000 
        whiteref = np.array(f['white_reference'])/10000
       
    
        hcube = np.swapaxes(hcube,-1,0).astype('float32')
        hcube = np.fliplr(hcube)
       
        wlens = f['hypercube'].attrs['wavelength_nm']

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

def calibrate_reflectance(hcube_raw, darkref, whiteref, eps=1e-6):
    H, W, B = hcube_raw.shape

    def _prepare(ref):
        if ref is None:
            return 0.0

        ref = np.asarray(ref)

        # Case 1: per-band vector (B,)
        if ref.ndim == 1:
            if ref.shape[0] != B:
                raise ValueError(f"1D ref length {ref.shape[0]} != n_bands {B}")
            return ref.reshape(1, 1, B)

        # Case 2: full cube already (H, W, B)
        if ref.ndim == 3:
            if ref.shape == (H, W, B):
                return ref
            if ref.shape == (1, W, B):
                return ref
            raise ValueError(f"3D ref shape {ref.shape} not compatible with cube {hcube_raw.shape}")

        # Case 3: 2D references
        if ref.ndim == 2:
            # (B, W) -> (1, W, B)
            if ref.shape == (B, W):
                ref2 = ref.T[None, :, :]      # (1, W, B)
                return ref2
            # (H, W) -> (H, W, 1)
            if ref.shape == (H, W):
                return ref[:, :, None]

            raise ValueError(f"2D ref shape {ref.shape} not compatible with cube {hcube_raw.shape}")

        raise ValueError(f"Unsupported ref ndim={ref.ndim} and shape={ref.shape}")

    dark = _prepare(darkref)
    white = _prepare(whiteref)

    num = hcube_raw - dark
    denom = white - dark

    R = num / (denom + eps)
    R = np.clip(R, 0, 1)

    return R

def make_rgb(hcube, band_ids=(60, 108, 163)):
    H, W, B = hcube.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    for i, b in enumerate(band_ids):
        v = hcube[:, :, b]
        lo, hi = np.percentile(v, (2, 98))  
        rgb[:, :, i] = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)

    return rgb


def build_foreground_mask(hcube, min_size=500, otsu_factor=0.50, use_watershed=True):
    """
    PCA_original-style mask, but with a relaxed Otsu threshold.
    """

    mean_img = hcube.mean(axis=2)

    thresh = threshold_otsu(mean_img)
    binary = mean_img > (otsu_factor * thresh)

    binary = remove_small_objects(binary, min_size=min_size)

    if not use_watershed:
        return binary

    dist = ndi.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=10, labels=binary)

    markers = np.zeros_like(binary, dtype=int)
    if len(coords) > 0:
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    markers = ndi.label(markers)[0]
    labels = watershed(-dist, markers, mask=binary)

    return labels > 0