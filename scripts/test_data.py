import argparse
from data import build_dataset, build_dataloader

def test_data():
    parser = argparse.ArgumentParser()
    opts = parser.parse_args([])
    opts.dataset_root = "./non_existent_folder"
    opts.dataset_name = "image_folder"
    
    print("Testing Dataset and DataLoader Builders...")
    train_dataset = build_dataset(opts, is_training=True)
    val_dataset = build_dataset(opts, is_training=False)
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    
    train_loader = build_dataloader(opts, is_training=True)
    if train_loader is None:
        print("DataLoader correctly returned None for empty dataset.")
        
    print("Data pipeline test passed!")

if __name__ == "__main__":
    test_data()
