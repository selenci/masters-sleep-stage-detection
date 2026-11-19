import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_center import EEGDataset, EEGTransformer
from focal_loss import FocalLoss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "../Data/"
TRAIN_PATIENTS = [
'PROCESSED_HFO/01', 'PROCESSED_HFO/02', 'PROCESSED_HFO/03', 'PROCESSED_HFO/04', 'PROCESSED_HFO/05', 'PROCESSED_HFO/06', 'PROCESSED_HFO/07', 'PROCESSED_HFO/08', 'PROCESSED_HFO/09', 'PROCESSED_HFO/11', 'PROCESSED_HFO/12', 'PROCESSED_HFO/13', 'PROCESSED_HFO/14', 'PROCESSED_HFO/15', 'PROCESSED_HFO/16', 'PROCESSED_HFO/17', 'PROCESSED_HFO/18', 'PROCESSED_HFO/19', 'PROCESSED_HFO/20', 'PROCESSED_HFO/21',
]

VALIDATION_PATIENTS = [
'PROCESSED_HFO/22', 'PROCESSED_HFO/23', 'PROCESSED_HFO/24', 'PROCESSED_HFO/25', 'PROCESSED_HFO/26',
]

TEST_PATIENTS = [
'PROCESSED_CAP_2/ins9',
'PROCESSED_CAP_2/n11',
'PROCESSED_CAP_2/narco5',
'PROCESSED_CAP_2/nfle38', 'PROCESSED_CAP_2/nfle39', 'PROCESSED_CAP_2/nfle40',
'PROCESSED_CAP_2/plm10',
'PROCESSED_CAP_2/rbd21', 'PROCESSED_CAP_2/rbd22',
'PROCESSED_AHD/EPCTL27', 'PROCESSED_AHD/EPCTL28', 'PROCESSED_AHD/EPCTL29',
'PROCESSED_HFO/27', 'PROCESSED_HFO/28', 'PROCESSED_HFO/29', 'PROCESSED_HFO/30',
]
DATASET_GROUPS = {"PROCESSED_CAP_2": "Adult", "PROCESSED_AHD": "Adult", "PROCESSED_HFO": "Pediatric"}

BATCH_SIZE = 16
LEARNING_RATE = 0.0000015
WEIGHT_DECAY = 1e-3
EPOCHS = 25
EMB_SIZE = 32
NUM_CLASSES = 5
print(f"Using EMB_SIZE={EMB_SIZE}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}, EPOCHS={EPOCHS}", flush=True)

# --- Dataset and DataLoader ---
train_dataset = EEGDataset(
    patients=TRAIN_PATIENTS,
    data_dir=DATA_DIR,
    group_maps=DATASET_GROUPS,
    is_train=True
)
train_mean, train_std = train_dataset.get_mean_std()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = EEGDataset(
    patients=VALIDATION_PATIENTS,
    data_dir=DATA_DIR,
    group_maps=DATASET_GROUPS,
)
validation_dataset.set_mean_std(train_mean, train_std)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for validation

height, width = train_dataset[0][0].shape[2], train_dataset[0][0].shape[3]
print(f"Input spectrogram dimensions: Height={height}, Width={width}", flush=True)

test_dataset = EEGDataset(
    patients=TEST_PATIENTS,
    data_dir=DATA_DIR,
    group_maps=DATASET_GROUPS,
)
test_dataset.set_mean_std(train_mean, train_std)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for evaluation

# --- Model, Loss, and Optimizer ---
model = EEGTransformer(
    emb_size=EMB_SIZE,
    num_classes=NUM_CLASSES,
    num_channels=6,
    height=height,
    width=width
).to(device)
model.load_state_dict(torch.load('best_model.pth'))

for param in model.cnn_embed.parameters():
    param.requires_grad = False

criterion = FocalLoss(task_type='multi-class', alpha=None, gamma=2.0, reduction='mean', num_classes=NUM_CLASSES)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),  # only unfrozen params
    lr=LEARNING_RATE,  # lower than pretraining
    weight_decay=WEIGHT_DECAY
)


PATIENCE = 8
current_patience = 0
best_val_loss = float('inf')
best_epoch = 0

def evaluate_and_print_cm(model, data_loader, device, class_names):
    """
    Evaluates the model, printing and plotting the confusion matrix for each data source and a joint matrix.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Use defaultdict to easily handle new data sources
    all_preds_by_source = defaultdict(list)
    all_labels_by_source = defaultdict(list)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for segments, labels, masks, sources in data_loader:
            segments, labels, masks, groups = segments.to(device), labels.to(device), masks.to(device)
            outputs = model(segments, masks)
            _, predicted = torch.max(outputs.data, 1)
            
            predicted_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(predicted_np)
            all_labels.extend(labels_np)

            # Store predictions and labels for each source
            for i in range(len(sources)):
                source = sources[i]
                all_preds_by_source[source].append(predicted_np[i])
                all_labels_by_source[source].append(labels_np[i])

    # Calculate and print metrics for each source
    for source in sorted(all_preds_by_source.keys()):
        source_cm = confusion_matrix(all_labels_by_source[source], all_preds_by_source[source])
        source_accuracy = np.trace(source_cm) / np.sum(source_cm)
        print(f"--- Confusion Matrix for source: {source} ---")
        print(source_cm)
        print(f"Accuracy for source {source}: {source_accuracy:.4f}\n")

    # Compute and print the joint confusion matrix
    joint_cm = confusion_matrix(all_labels, all_preds)
    joint_accuracy = np.trace(joint_cm) / np.sum(joint_cm)
    print("--- Joint Confusion Matrix ---")
    print(joint_cm)
    print(f"Joint Accuracy: {joint_accuracy:.4f}", flush=True)

    # Plot all confusion matrices in one figure
    sources = sorted(all_preds_by_source.keys())
    num_sources = len(sources)
    fig, axes = plt.subplots(1, num_sources + 1, figsize=(10 * (num_sources + 1), 8))
    
    # Plot individual source confusion matrices
    for i, source in enumerate(sources):
        cm = confusion_matrix(all_labels_by_source[source], all_preds_by_source[source])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - Source: {source}')
        axes[i].set_ylabel('Actual Label')
        axes[i].set_xlabel('Predicted Label')

    # Plot the joint confusion matrix
    sns.heatmap(joint_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[num_sources])
    axes[num_sources].set_title('Joint Confusion Matrix')
    axes[num_sources].set_ylabel('Actual Label')
    axes[num_sources].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('all_confusion_matrices.png')
    plt.show()



for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
    for segments, labels, masks, sources in train_loader:
        segments, labels, masks, groups = segments.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(segments, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for segments, labels, masks, sources in validation_loader:
            segments, labels, masks, groups = segments.to(device), labels.to(device), masks.to(device)
            outputs = model(segments, masks, groups)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(validation_loader)
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        current_patience = 0
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'finetuned_best_model.pth')
        print(f"Validation loss decreased. Saving new best model.", flush=True)

    else:
        current_patience += 1
        if current_patience >= PATIENCE:
            print("Early stopping triggered.", flush=True)
            break

print("Training finished.")

# --- Evaluation on Test Set with the Best Model ---
print("\nLoading the best model for final evaluation on the test set...")
model.load_state_dict(torch.load('finetuned_best_model.pth'))

if len(test_dataset) > 0:
    evaluate_and_print_cm(model, test_loader, device, ['W', 'N1', 'N2', 'N3', 'R'])
else:
    print("Warning: No test data found. Skipping final evaluation.")