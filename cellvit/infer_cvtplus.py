
import os
import subprocess
import pandas as pd
from glob import glob
from natsort import natsorted
# from eval_util import evaluate_cellvit

DATASETS = [
    "consep",
    "cpm15",
    "cpm17",
    "cryonuseg",
    "lizard",
    "lynsec_he",
    "lynsec_ihc",
    "monusac",
    "monuseg",
    "nuclick",
    "nuinsseg",
    "pannuke",
    "puma",
    "srsanet",
    "tnbc",
]

CVTPP_CP = [
    # 'Virchow-x40-AMP',
    'SAM-H-x40-AMP',
    # '256-x40-AMP'
]


def run_inference(model_dir, input_dir, output_dir):
    for dataset in DATASETS:
        data_dir = os.path.join(input_dir, dataset)
        samples = list(natsorted(glob(os.path.join(data_dir, '*.tiff'))))
        files = {"path": samples,
                 "slide_mpp": [0.25 for i in range(len(samples))],
                 "magnification": [40 for i in range(len(samples))]
                }
        if len(files["path"]) == 0:
            print(f"No valid input path given for {dataset}.")
            continue
        filelist_df = pd.DataFrame(files)
        os.makedirs(os.path.join(input_dir, "file_lists"), exist_ok=True)
        csv_filelist = os.path.join(input_dir, "file_lists", f"{dataset}_filelist.csv")
        filelist_df.to_csv(csv_filelist, index=False)
        for checkpoint in CVTPP_CP:
            checkpoint_path = os.path.join(model_dir, f"CellViT-{checkpoint}.pth")
            output_path = os.path.join(output_dir, dataset, checkpoint)
            if os.path.exists(output_path):
                if len(os.listdir(output_path)) > 1:
                    continue
            os.makedirs(output_path, exist_ok=True)
            args = [
                "--model",
                f"{checkpoint_path}",
                "--outdir",
                f"{output_path}",
                "process_dataset",
                "--filelist",
                f"{csv_filelist}"
            ]
            command = [
                "python3",
                "/user/titus.griebel/u12649/CellViT-plus-plus/cellvit/detect_cells.py",
            ] + args
            print(f"Running inference with CellViT-plus-plus {checkpoint} model on {dataset} dataset...")
            subprocess.run(command)
            for file in glob(os.path.join(output_path, '*json')):
                os.remove(file)    
            print(f"Successfully ran inference with CellViT {checkpoint} model on {dataset} dataset")


def main():
    run_inference(
        "/mnt/lustre-grete/usr/u12649/models/cellvit_plusplus/checkpoints",
        "/mnt/lustre-grete/usr/u12649/data/cvtplus/preprocessed",
        "/mnt/lustre-grete/usr/u12649/models/cellvit_plusplus/inference/",
    )


if __name__ == "__main__":
    main()
