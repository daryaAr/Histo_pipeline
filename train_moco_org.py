import os
import logging
import csv
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from Loss import ContrastiveLoss  
from moco_model_encoder import MoCoV2Encoder  
import time
import math
from moco_data_loader import MoCoTileDataset, TwoCropsTransform, get_moco_v2_augmentations



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories for saving models, plots, and CSV results
#MODEL_SAVE_DIR = "/home/darya/Histo_pipeline/Moco_Original_models"
SECOND_SAVE_DIR = "/mnt/nas7/data/Personal/Darya/saved_models/moco_50_256"
CHECKPOINT_SAVE_DIR = "/mnt/nas7/data/Personal/Darya/Checkpoints/moco_50_256" 
PLOT_SAVE_DIR = "/home/darya/Histo_pipeline/Loss_curve_plot"
CSV_SAVE_PATH = "/home/darya/Histo_pipeline/MoCo_org.csv"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)

# Create the CSV file if it doesn't exist
if not os.path.exists(CSV_SAVE_PATH):
    with open(CSV_SAVE_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Epoch", "Batch Size", "ResNet Type", "Average Training Loss", 
            "Training Time", "Metric Type", "Number of Epochs"
        ])
    logger.info(f"CSV file created at {CSV_SAVE_PATH}")


torch.cuda.empty_cache()

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def update_template_csv(csv_path, batch_size, temperature, avg_loss, train_time, num_epochs):
    """
    Update the CSV file with training details.
    """
    csv_headers = [
        "Batch Size", "Temperature", "Average Training Loss", 
        "Training Time", "Metric Type", "Number of Epochs"
    ]
    row_data = [
        batch_size, temperature, avg_loss, train_time, 
        "ContrastiveLoss", num_epochs
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow(row_data)


# Learning Rate Scheduler (Cosine with Warmup) - Manual
def adjust_learning_rate(optimizer, epoch, base_lr, total_epochs):
    """
    MoCo v2: Linear warmup for first 10 epochs, then cosine decay.
    """
    warmup_epochs = 10  
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs  # Linear warmup
    else:
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))  # Cosine decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # Apply LR to optimizer
    
    return lr  # Return LR for logging

def save_checkpoint(epoch, model, optimizer, scaler, base_lr, checkpoint_path, best=False):
    """ Save model, optimizer, scaler, and learning rate state. """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'learning_rate': base_lr  # Save base LR
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    if best:
        best_path = os.path.join(CHECKPOINT_SAVE_DIR, "moco_best_model_256_101.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"New best model saved: {best_path}")

def train_moco(
    tile_csv_path,
    model=None,
    batch_size=128,
    epochs=100,
    learning_rate=0.003,
    temperature=0.07,
    csv_path=CSV_SAVE_PATH,
    device="cuda",
    resnet_type=None,
    resume_checkpoint=None
):
    logger.info("Starting MoCo training with Mixed Precision")
    start_time = datetime.now()

    
    # DataLoader using MoCo-style augmentations
    logger.info("Initializing DataLoader...")
    train_transform = TwoCropsTransform(get_moco_v2_augmentations())
    train_dataset = MoCoTileDataset(csv_path=tile_csv_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    logger.info(f"Training DataLoader loaded with {len(train_loader)} batches.")


    # Optimizer and 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
   
    # Loss function
    criterion = ContrastiveLoss()

    # Resume from a checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        learning_rate = checkpoint['learning_rate']  # Restore LR
        logger.info(f"Resuming training from epoch {start_epoch}, LR: {learning_rate}")


    # Move model to device
    model.to(device)

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    # Training variables
    epoch_losses = []
        
    best_loss = float("inf")
   
    # TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs} started.")
        model.train()
        running_loss = 0.0
        batch_time = []
        num_batches = 0

         # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, learning_rate, epochs)
        logger.info(f"Learning rate adjusted to: {lr:.6f}")

        for batch_idx, (images_q, images_k) in enumerate(train_loader):
            start_time_batch = time.perf_counter()
    
            images_q = images_q.to(device, non_blocking=True)
            images_k = images_k.to(device, non_blocking=True)

            # Forward pass
            with torch.amp.autocast(device_type="cuda"):
                q, k = model(images_q, images_k)  
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
            num_batches += 1

            #if batch_idx + 1 == 20:
             #   break

        avg_batch_time = sum(batch_time)/len(batch_time)
        avg_epoch_loss = running_loss / num_batches if num_batches > 0 else float("inf")
        epoch_losses.append(avg_epoch_loss)
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        logger.info(f"Epoch {epoch + 1} completed with Loss: {avg_epoch_loss:.4f}")
        logger.info(f"Average time of running a batch: {avg_batch_time:.6f} seconds ")
        
       

        # Save checkpoint
        checkpoint_path = os.path.join(SECOND_SAVE_DIR, f"moco_org_{batch_size}_{resnet_type}_{epoch}.pth")
        save_checkpoint(epoch, model, optimizer, scaler, learning_rate, checkpoint_path)

        # Save every 5 epochs and best model
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_SAVE_DIR, f"moco_checkpoint_epoch_{epoch+1}_256_50.pth")
            save_checkpoint(epoch, model, optimizer, scaler, learning_rate, checkpoint_path)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_checkpoint(epoch, model, optimizer, scaler, learning_rate, checkpoint_path, best=True)




    writer.close()


    elapsed_time = datetime.now() - start_time
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    update_template_csv(csv_path, batch_size, temperature, avg_loss, str(elapsed_time), epochs)

    # Plot and save training loss curve
    logger.info("Plotting and saving training loss curve...")
    train_plot_path = os.path.join(PLOT_SAVE_DIR, f"MOCO_org_loss_curve_256_50.png")
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

    

if __name__ == "__main__":
    resnet_type = "resnet50"
    moco_model = MoCoV2Encoder(base_encoder=resnet_type, output_dim=128, queue_size=65536)  # Reduce from 65536 32768
   

    # Path to tile_path.csv (contains all  train image paths)
    tile_csv_path = "/home/darya/Histo/Histo_pipeline_csv/train_path.csv"
   
    train_moco(
        tile_csv_path=tile_csv_path,
        model=moco_model,
        batch_size=256,
        epochs=50,
        learning_rate=0.003,
        temperature=0.07,
        csv_path=CSV_SAVE_PATH,
        device=device,
        resnet_type=resnet_type
    )

 