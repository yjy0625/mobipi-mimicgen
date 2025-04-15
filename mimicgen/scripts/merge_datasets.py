"""
This script merges multiple HDF5 datasets stored in subdirectories under a common root directory.
Each subdirectory contains a `demo/demo.hdf5` file with demos in the format `demo_0`, `demo_1`, etc.
The merged output renumbers all demos sequentially, copies metadata from the first dataset, and
preserves per-demo attributes. Associated `mg_config.json` files are also copied.

Usage:
    python merge_datasets.py --root /path/to/root_folder \
                             --subdirs mg_nav1 mg_nav2 \
                             --output merged_dataset_name

The merged dataset will be saved to:
    /path/to/root_folder/merged_dataset_name/demo/demo.hdf5
    with mg_config.json files renamed and copied into the same folder.
"""
import os
import argparse
import h5py
import shutil
from tqdm import tqdm

def merge_datasets(root, subdirs, output_name):
    output_dir = os.path.join(root, output_name, "demo")
    os.makedirs(output_dir, exist_ok=True)

    output_h5_path = os.path.join(output_dir, "demo.hdf5")
    output_h5 = h5py.File(output_h5_path, "w")
    output_data = output_h5.create_group("data")

    demo_counter = 0
    first_attrs_copied = False

    for subdir in subdirs:
        demo_path = os.path.join(root, subdir, "demo")
        h5_path = os.path.join(demo_path, "demo.hdf5")
        json_path = os.path.join(root, subdir, "mg_config.json")

        if not os.path.exists(h5_path):
            print(f"Skipping {subdir}, no demo.hdf5 found.")
            continue

        with h5py.File(h5_path, "r") as f:
            src_data = f["data"]
            demo_keys = sorted(src_data.keys(), key=lambda x: int(x.split("_")[1]))

            # Copy top-level data attrs only once
            if not first_attrs_copied:
                for k, v in src_data.attrs.items():
                    output_data.attrs[k] = v
                first_attrs_copied = True

            for demo_key in tqdm(demo_keys, desc=f"Merging {subdir}"):
                new_demo_key = f"demo_{demo_counter}"
                f.copy(f["data"][demo_key], output_data, name=new_demo_key)

                # Copy demo-level attrs
                for attr_key, attr_val in f["data"][demo_key].attrs.items():
                    output_data[new_demo_key].attrs[attr_key] = attr_val

                demo_counter += 1

        # Copy mg_config.json into output_dir with renamed name
        if os.path.exists(json_path):
            shutil.copy(json_path, os.path.join(output_dir, f"{subdir}_mg_config.json"))

    output_h5.close()
    print(f"Merge complete. Total demos: {demo_counter}")
    print(f"Merged dataset stored at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root folder containing subdirectories")
    parser.add_argument("--subdirs", nargs="+", required=True, help="Subdirectory names to merge")
    parser.add_argument("--output", type=str, required=True, help="Name for the output merged dataset")

    args = parser.parse_args()
    merge_datasets(args.root, args.subdirs, args.output)
