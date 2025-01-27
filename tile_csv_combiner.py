import pandas as pd
from pathlib import Path

# Define directories with the correct paths
luad_dir = Path('/mnt/nas7/data/Personal/Valentin/histopath/tiles_14jan25/tcga_luad')
lusc_dir = Path('/mnt/nas7/data/Personal/Valentin/histopath/tiles_14jan25/tcga_lusc')
cptac_luad_dir = Path('/mnt/nas7/data/Personal/Valentin/histopath/tiles_14jan25/cptac_luad')
cptac_lusc_dir = Path('/mnt/nas7/data/Personal/Valentin/histopath/tiles_14jan25/cptac_lusc')

# Output file directory and file name
output_dir = Path("/home/darya/Histo/Histo_pipeline_csv")
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
output_file = output_dir / "tile_selection_total_histopath.csv"

# List to hold data from each __tiling_results.csv file
data_frames = []

# Function to count folders in a directory
def count_folders(directory):
    return sum(1 for item in directory.iterdir() if item.is_dir())

# Function to process directories
def process_directory(directory):
    for folder in directory.iterdir():
        if folder.is_dir():
            print(f"Processing WSI folder: {folder.name}")
            csv_file = folder / f"{folder.name}__tiling_results.csv"  # Directly look for the CSV file
            if csv_file.is_file():
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Remove rows where 'keep' column is 0
                if 'keep' in df.columns:
                    original_count = len(df)
                    df = df[df['keep'] != 0]
                    filtered_count = len(df)
                    print(f"Filtered {original_count - filtered_count} rows where 'keep' == 0 in WSI folder: {folder.name}")
                
                # Add WSI column with the folder name
                df['WSI'] = folder.name
                data_frames.append(df)
                print(f"Finished processing WSI folder: {folder.name}")
            else:
                print(f"No __tiling_results.csv found for WSI folder: {folder.name}")

# Count and print the number of folders in each directory
luad_folder_count = count_folders(luad_dir)
lusc_folder_count = count_folders(lusc_dir)
cptac_luad_folder_count = count_folders(cptac_luad_dir)
cptac_lusc_folder_count = count_folders(cptac_lusc_dir)

print(f"Number of folders in LUAD directory: {luad_folder_count}")
print(f"Number of folders in LUSC directory: {lusc_folder_count}")
print(f"Number of folders in CPTAC_LUAD directory: {cptac_luad_folder_count}")
print(f"Number of folders in CPTAC_LUSC directory: {cptac_lusc_folder_count}")

# Process each directory
process_directory(luad_dir)
process_directory(lusc_dir)
process_directory(cptac_luad_dir)
process_directory(cptac_lusc_dir)

# Concatenate all data frames and save to a single CSV file
if data_frames:
    total_df = pd.concat(data_frames, ignore_index=True)
    total_df.to_csv(output_file, index=False)
    print(f"Consolidated CSV saved to {output_file}")
else:
    print("No __tiling_results.csv files found.")