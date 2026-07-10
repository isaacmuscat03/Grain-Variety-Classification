from pathlib import Path
import pandas as pd
import subprocess, os
import h5py
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, remove_small_holes, closing, opening, disk, binary_erosion, disk
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.ndimage import median_filter
from skimage import exposure

def load_data(
    mode="wsl",
    dataset_path=None,
    windows_drive="E:",
    windows_path=r"E:\HSI_Dataset_2\Elements\data",
    link_name="data_external",
    varieties=("barley", "corn", "flax"),
):

    valid_modes = {"wsl", "direct"}

    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {sorted(valid_modes)}, got {mode!r}."
        )

    # Original WSL workflow
    if mode == "wsl":
        mount_path = Path(
            f"/mnt/{windows_drive[0].lower()}"
        )

        print(
            f"Mounting {windows_drive} into {mount_path} ..."
        )

        subprocess.run(
            ["sudo", "mkdir", "-p", str(mount_path)],
            check=True,
        )

        mount_result = subprocess.run(
            [
                "sudo",
                "mount",
                "-t",
                "drvfs",
                windows_drive,
                str(mount_path),
            ],
            capture_output=True,
            text=True,
        )

        if mount_result.returncode != 0:
            error_message = mount_result.stderr.strip()


            already_mounted = (
                "already mounted" in error_message.lower()
                or "mount point is busy" in error_message.lower()
            )

            if not already_mounted:
                raise RuntimeError(
                    f"Failed to mount {windows_drive}: "
                    f"{error_message}"
                )

            print(
                f"[OK] {windows_drive} appears to be already mounted."
            )

        # Convert:
        # E:\HSI_Dataset_2\Elements\data
        # into:
        # /mnt/e/HSI_Dataset_2/Elements/data
        drive_prefix = windows_drive.rstrip(":")

        normalized_windows_path = (
            str(windows_path)
            .replace("\\", "/")
        )

        if not normalized_windows_path.lower().startswith(
            drive_prefix.lower() + ":"
        ):
            raise ValueError(
                "windows_path does not begin with windows_drive:\n"
                f"windows_drive={windows_drive!r}\n"
                f"windows_path={windows_path!r}"
            )

        relative_windows_path = (
            normalized_windows_path
            .split(":", 1)[1]
            .lstrip("/")
        )

        resolved_dataset_path = (
            mount_path / relative_windows_path
        )

        if not resolved_dataset_path.exists():
            print(
                f"ERROR: Dataset not found at "
                f"{resolved_dataset_path}."
            )
            print(
                f"Contents currently visible under {mount_path}:"
            )

            os.system(f'ls -la "{mount_path}"')

            raise FileNotFoundError(
                "Fix windows_drive/windows_path and rerun."
            )

        print(
            f"[OK] Found dataset: {resolved_dataset_path}"
        )

        project_root = Path.cwd()
        link_path = project_root / link_name

        if link_path.exists() or link_path.is_symlink():
            print(f"Removing old link {link_path}")

            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            elif link_path.is_dir():
                raise IsADirectoryError(
                    f"{link_path} is a real directory rather than "
                    "a symlink. It was not removed automatically."
                )

        link_path.symlink_to(
            resolved_dataset_path,
            target_is_directory=True,
        )

        print(
            f"[OK] Linked {link_path} -> "
            f"{resolved_dataset_path}"
        )

        extensions = {
            ".hdf5",
            ".h5",
            ".hdr",
            ".tif",
            ".tiff",
        }

        found_examples = list(
            itertools.islice(
                (
                    path
                    for path in link_path.rglob("*")
                    if path.suffix.lower() in extensions
                ),
                10,
            )
        )

        if found_examples:
            print("Sample files:")

            for filepath in found_examples:
                print(
                    "  ",
                    filepath.relative_to(link_path),
                )
        else:
            print(
                "No .hdf5/.h5/.hdr/.tif files found yet — "
                "check deeper folders."
            )

        root = Path(link_name)
        hr_root = root / "raw" / "FX10"

        outdir = (
            hr_root.parent.parent
            / "processed"
            / "quickrun"
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # direct/local workflow
    else:
        if dataset_path is None:
            raise ValueError(
                "dataset_path must be provided when mode='direct'."
            )

        resolved_dataset_path = (
            Path(dataset_path)
            .expanduser()
            .resolve()
        )

        if not resolved_dataset_path.exists():
            raise FileNotFoundError(
                "Dataset not found at:\n"
                f"{resolved_dataset_path}"
            )

        if not resolved_dataset_path.is_dir():
            raise NotADirectoryError(
                "dataset_path is not a directory:\n"
                f"{resolved_dataset_path}"
            )

        print(
            f"[OK] Found dataset: {resolved_dataset_path}"
        )

        # The direct path itself replaces data_external.
        root = resolved_dataset_path
        hr_root = root / "raw" / "FX10"

        # Keep the same derived output-directory behaviour.
        outdir = (
            hr_root.parent.parent
            / "processed"
            / "quickrun"
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        extensions = {
            ".hdf5",
            ".h5",
            ".hdr",
            ".tif",
            ".tiff",
        }

        found_examples = list(
            itertools.islice(
                (
                    path
                    for path in root.rglob("*")
                    if path.suffix.lower() in extensions
                ),
                10,
            )
        )

        if found_examples:
            print("Sample files:")

            for filepath in found_examples:
                try:
                    shown_path = filepath.relative_to(root)
                except ValueError:
                    shown_path = filepath

                print("  ", shown_path)
        else:
            print(
                "No .hdf5/.h5/.hdr/.tif files found yet — "
                "check dataset_path."
            )

    # Shared original loading/splitting functionality
    print("ROOT:    ", root.resolve())
    print("Basepath:", hr_root.resolve())
    print("OUTDIR:  ", outdir.resolve())

    if not hr_root.exists():
        raise FileNotFoundError(
            "Expected FX10 folder was not found:\n"
            f"{hr_root}\n\n"
            "The supplied dataset root must contain raw/FX10."
        )

    # Preserve the original preliminary DataFrame creation
    df = pd.DataFrame(
        {
            "filepath_FX10": list(
                Path(f"{hr_root}").rglob("**/*.hdf5")
            )
        }
    )

    df["sample_name"] = df["filepath_FX10"].apply(
        lambda path: path.stem
    )

    # Find all .hdf5 files
    all_files = list(
        hr_root.rglob("*.hdf5")
    )

    print(
        "Total .hdf5 files found:",
        len(all_files),
    )

    if len(all_files) == 0:
        raise FileNotFoundError(
            "No .hdf5 files were found under:\n"
            f"{hr_root}"
        )

    df_files = pd.DataFrame(
        {"filepath_FX10": all_files}
    )

    df_files["sample_name"] = (
        df_files["filepath_FX10"]
        .apply(lambda path: path.stem)
    )

    def resolve_valid_hdf5(path):
        """
        Return a Path when the file is readable as HDF5,
        otherwise return None.
        """
        path = Path(path)

        try:
            with h5py.File(path, "r"):
                return path
        except Exception:
            return None

    df_files["resolved_path"] = (
        df_files["filepath_FX10"]
        .apply(resolve_valid_hdf5)
    )

    df_valid = df_files[
        df_files["resolved_path"].notna()
    ].copy()

    df_invalid = df_files[
        df_files["resolved_path"].isna()
    ].copy()

    print(
        "Valid HDF5 files:",
        len(df_valid),
    )

    print(
        "Invalid/unreadable files:",
        len(df_invalid),
    )

    if not df_invalid.empty:
        print("\nUnreadable sample_names:")
        print(
            df_invalid["sample_name"].tolist()
        )

    if df_valid.empty:
        raise RuntimeError(
            "No valid HDF5 files were found."
        )

    # Create image-level train/valid/test split
    df_valid["variety"] = (
        df_valid["sample_name"]
        .str.split("_")
        .str[0]
    )

    df_valid["batch"] = (
        df_valid["sample_name"]
        .str.split("_")
        .str[-1]
    )

    df_valid["size"] = (
        df_valid["batch"].str[0]
    )

    df_valid["rep"] = pd.to_numeric(
        df_valid["batch"].str[1:],
        errors="raise",
    ).astype(int)

    varieties = list(varieties)

    df_valid = df_valid[
        df_valid["variety"].isin(varieties)
    ].copy()

    def assign_split(group):
        group = group.sort_values("rep").copy()

        if len(group) != 5:
            print(
                "Warning: expected 5 images, got",
                len(group),
                "for",
                group[
                    ["variety", "size"]
                ].iloc[0].to_dict(),
            )

        group["split"] = "unused"

        group.iloc[
            :3,
            group.columns.get_loc("split"),
        ] = "train"

        group.iloc[
            3:4,
            group.columns.get_loc("split"),
        ] = "valid"

        group.iloc[
            4:5,
            group.columns.get_loc("split"),
        ] = "test"

        return group

    files_split = (
        df_valid
        .groupby(
            ["variety", "size"],
            group_keys=False,
        )
        .apply(assign_split)
        .reset_index(drop=True)
    )

    train_files = files_split[
        files_split["split"] == "train"
    ].copy()

    valid_files = files_split[
        files_split["split"] == "valid"
    ].copy()

    test_files = files_split[
        files_split["split"] == "test"
    ].copy()

    # Preserve the original final train metadata assignments
    train_files.loc[:, "variety"] = (
        train_files["sample_name"]
        .str.split("_")
        .str[0]
    )

    train_files.loc[:, "batch"] = (
        train_files["sample_name"]
        .str.split("_")
        .str[-1]
    )

    train_files.loc[:, "size"] = (
        train_files["batch"].str[0]
    )

    train_files.loc[:, "rep"] = (
        train_files["batch"]
        .str[1:]
        .astype(int)
    )

    return (
        train_files,
        valid_files,
        test_files,
        files_split,
    )

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