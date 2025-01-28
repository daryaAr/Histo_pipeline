import os
import logging
import csv
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from Loss import ContrastiveLoss  
from data_loader_histopath import get_loader  
from moco_model_encoder import MoCoV2Encoder 
from validation_utils import get_validation_loader, validate_model  
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories for saving models, plots, and CSV results
MODEL_SAVE_DIR = "/home/darya/Histo_pipeline/Moco_Original_models"
SECOND_SAVE_DIR = "/mnt/nas7/data/Personal/Darya/saved_models"
PLOT_SAVE_DIR = "/home/darya/Histo_pipeline/Loss_curve_plot"
CSV_SAVE_PATH = "/home/darya/Histo_pipeline/MoCo_org.csv"
VALIDATION_UMAP_DIR = "/home/darya/Histo_pipeline/UMAP"
VALIDATION_CSV_PATH = "/home/darya/Histo_pipeline/validation_org_moco.csv"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(VALIDATION_UMAP_DIR, exist_ok=True)

# Create the CSV file if it doesn't exist
if not os.path.exists(CSV_SAVE_PATH):
    with open(CSV_SAVE_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Epoch", "Batch Size", "ResNet Type", "Average Training Loss", 
            "Training Time", "Convergence Speed", "Metric Type", "Number of Epochs"
        ])
    logger.info(f"CSV file created at {CSV_SAVE_PATH}")

# Create validation CSV file with headers if it doesn't exist
if not os.path.exists(VALIDATION_CSV_PATH):
    with open(VALIDATION_CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Epoch", "Batch Size", "ResNet Type", "Contrastive Loss", 
            "Linear Accuracy", "Linear Std", "k-NN Accuracy", "k-NN Std"
        ])

torch.cuda.empty_cache()

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def update_template_csv(csv_path, batch_size, temperature, avg_loss, train_time, convergence_speed, num_epochs):
    """
    Update the CSV file with training details.
    """
    csv_headers = [
        "Batch Size", "Temperature", "Average Training Loss", 
        "Training Time", "Convergence Speed", "Metric Type", "Number of Epochs"
    ]
    row_data = [
        batch_size, temperature, avg_loss, train_time, 
        convergence_speed, "ContrastiveLoss + Accuracy Metrics", num_epochs
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow(row_data)

def apply_augmentations(center_images):
    """
    Apply data augmentations to the input images.
    """
    augmentation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    augmented_images = torch.stack([augmentation_pipeline(image) for image in center_images])
    return augmented_images

def train_moco(
    neighbors_csv,
    val_data_dir,
    val_batch_size,
    model,
    batch_size=128,
    epochs=100,
    learning_rate=0.001,
    patience=10,
    temperature=0.07,
    csv_path=CSV_SAVE_PATH,
    validation_csv_path=VALIDATION_CSV_PATH,
    device="cuda"
):
    logger.info("Starting MoCo training with Mixed Precision")
    start_time = datetime.now()

    # DataLoaders
    logger.info("Loading training DataLoader...")
    train_loader = get_loader(
        neighbors_csv=neighbors_csv,
        batch_size=batch_size,
        augment=False,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2
    )
    logger.info(f"Training DataLoader loaded with {len(train_loader)} batches.")

    logger.info("Loading validation DataLoader...")
    val_loader = get_validation_loader(val_data_dir, val_batch_size, augment=False, num_workers=8, pin_memory=False)
    logger.info("Validation DataLoader loaded.")

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    # Loss function
    criterion = ContrastiveLoss()

    # Move model to device
    model.to(device)

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    # Training variables
    best_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    save_path = os.path.join(MODEL_SAVE_DIR, "moco_v2_model_256_101.pth")
    epoch_losses = []
    validation_losses = []


    # TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs} started.")
        model.train()
        running_loss = 0.0
        batch_time = []

        for batch_idx, batch in enumerate(train_loader):
            start_time_batch = time.perf_counter()
            center_images, _, _, _, _ = batch
            logger.info(f"Processing Batch {batch_idx + 1}/{len(train_loader)}.")

            # Apply augmentations to center images
            center_images_aug = apply_augmentations(center_images)
            positive_images = apply_augmentations(center_images)
            logger.info("Augmentations created.")

            # Move data to device
            center_images_aug = center_images_aug.to(device, non_blocking=True)
            positive_images = positive_images.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                # Forward pass
                q, k = model(center_images_aug, positive_images)

                # Compute contrastive loss
                loss = criterion(q, k, model.queue)
                logger.info("Loss Calculated")

            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update queue
            model.update_queue(k)

            running_loss += loss.item()
            end_time_batch = time.perf_counter()
            step_time_batch = end_time_batch - start_time_batch
            batch_time.append(step_time_batch)

            logger.info(f"Batch {batch_idx+1} took {step_time_batch:.6f} seconds")
            logger.info(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            #if batch_idx + 1 == 100:
             #   break

        avg_batch_time = sum(batch_time)/len(batch_time)
        avg_epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        logger.info(f"Epoch {epoch + 1} completed with Loss: {avg_epoch_loss:.4f}")
        logger.info(f"Average time of running a batch: {avg_batch_time:.6f} seconds ")

        scheduler.step(avg_epoch_loss)

        # Save the model at the end of every epoch
        model_name = f"moco_org_{batch_size}_{model.base_encoder}_{epoch}.pth"
        save_path_1 = os.path.join(MODEL_SAVE_DIR, model_name)
        save_path_2 = os.path.join(SECOND_SAVE_DIR, model_name)

        torch.save(model.state_dict(), save_path_1)
        torch.save(model.state_dict(), save_path_2)
        logger.info(f"Model saved at: {save_path_1} and {save_path_2}")

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info(f"Starting validation at Epoch {epoch + 1}...")
            val_loss, linear_acc, linear_std, knn_acc, knn_std = validate_model(
                model,
                val_loader,
                criterion,
                epoch + 1,
                device,
                save_umap_dir=VALIDATION_UMAP_DIR,
                model_type="moco_org"
            )
            validation_losses.append(val_loss)  

            logger.info(
                f"Validation Results at Epoch {epoch + 1}: "
                f"Contrastive Loss={val_loss:.4f}, "
                f"Linear Acc={linear_acc:.2%} ± {linear_std:.2%}, "
                f"k-NN Acc={knn_acc:.2%} ± {knn_std:.2%}"
            )

            # Save validation metrics to CSV
            with open(VALIDATION_CSV_PATH, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)  
                csv_writer.writerow([
                    epoch + 1, batch_size, "ResNet101", val_loss, 
                    linear_acc, linear_std, knn_acc, knn_std
                ])

        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved with best loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    writer.close()

    # Final logging
    convergence_speed = best_epoch / epochs
    elapsed_time = datetime.now() - start_time
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    update_template_csv(csv_path, batch_size, temperature, avg_loss, str(elapsed_time), convergence_speed, epochs)

    # Plot and save training loss curve
    logger.info("Plotting and saving training loss curve...")
    train_plot_path = os.path.join(PLOT_SAVE_DIR, f"MOCO_org_loss_curve_256_101.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(train_plot_path)
    plt.close()
    logger.info(f"Training loss curve saved at {train_plot_path}")

    # Plot and save validation loss curve
    logger.info("Plotting and saving validation loss curve...")
    val_plot_path = os.path.join(PLOT_SAVE_DIR, f"validation_loss_org_moco_256_101.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(5, len(validation_losses) * 5 + 1, 5), validation_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(val_plot_path)
    plt.close()
    logger.info(f"Validation loss curve saved at {val_plot_path}")

if __name__ == "__main__":
    resnet_type = "resnet101"
    moco_model = MoCoV2Encoder(base_encoder=resnet_type, output_dim=128, queue_size=32768)  # Reduce from 65536

    # Paths to training and validation data
    train_neighbors_csv = "/home/darya/Histo/Histo_pipeline_csv/updated_neighbor_results.csv"
    val_data_dir = "/mnt/nas7/data/Personal/Darya/kaggle/lung_colon_image_set/lung_image_sets"

    train_moco(
        neighbors_csv=train_neighbors_csv,
        val_data_dir=val_data_dir,
        val_batch_size=256,
        model=moco_model,
        batch_size=256,
        epochs=50,
        learning_rate=0.001,
        patience=10,
        device=device,
    )