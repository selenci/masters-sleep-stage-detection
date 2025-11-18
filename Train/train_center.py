import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mne
import glob
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from focal_loss import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EEGDataset(Dataset):
    def __init__(self, patients, data_dir, group_maps, is_train=False):
        self.is_train = is_train
        self.files = []
        self.labels = []
        self.source_dataset = []

        self.LABEL_MAP = {
            "W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 3, "R": 4,
            "N1":1, "N2":2, "N3":3,
        }
        self.group_maps = group_maps  # e.g., {"PROCESSED_CAP_2": "Adult", "PROCESSED_AHD": "Adult", "PROCESSED_HFO": "Pediatric"}

        self.class_names = list(self.LABEL_MAP.keys())
        self.files_per_window = 10

        for patient_id_str in patients:
            source_dataset = patient_id_str.split('/')[0]
            folder = os.path.join(data_dir, f"{patient_id_str}")
            if not os.path.isdir(folder):
                print(f"Warning: Directory {folder} does not exist. Skipping.")
                continue
            
            npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")), 
                               key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            # *** OPTIMIZATION START ***
            # 1. Pre-load all labels for the current patient into a dictionary.
            # This reads from the disk only ONCE per file for this patient.
            labels_map = {}
            for npy_file in npy_files:
                base_name = os.path.splitext(npy_file)[0]
                label_path = base_name + "-stage.txt"
                with open(label_path, "r") as f:
                    label_str = f.read().strip()
                if label_str in self.LABEL_MAP:
                    labels_map[npy_file] = self.LABEL_MAP[label_str]
            # *** OPTIMIZATION END ***

            # Create overlapping windows with a stride of 1
            for i in range(len(npy_files) - self.files_per_window + 1):
                window_files = npy_files[i : i + self.files_per_window]
                
                # The center file determines the label
                center_file = window_files[self.files_per_window // 2]
                
                # *** USE THE FAST LOOKUP ***
                # Check if all files in the window and the center file have valid labels
                if all(f in labels_map for f in window_files):
                    center_label = labels_map[center_file]
                    self.files.append(window_files)
                    self.source_dataset.append(source_dataset)
                    self.labels.append(center_label)

    def __len__(self):
        return len(self.files)

    def get_mean_std(self):
        keys =  list(set(self.group_maps.values()))
        total_sum = {k:0.0 for k in keys}
        total_sq_sum = {k:0.0 for k in keys}
        total_count = {k:0 for k in keys}

        file_to_age_group_map = {}
        for i, window_files in enumerate(self.files):
            source_dataset = self.source_dataset[i]
            age_group = self.group_maps.get(source_dataset, "Adult")
            for file_path in window_files:
                file_to_age_group_map[file_path] = age_group

        for path, age_group in file_to_age_group_map.items():
            x_spec = np.load(path)
            x = x_spec.ravel()  # flatten, no copy
            total_sum[age_group] += x.sum()
            total_sq_sum[age_group] += np.square(x).sum()
            total_count[age_group] += x.size

        mean = {k: total_sum[k] / total_count[k] for k in keys}
        # E[x^2] - (E[x])^2
        var = {k: (total_sq_sum[k] / total_count[k]) - mean[k]**2 for k in keys}
        std = {k: np.sqrt(var[k]) for k in keys}

        self.mean, self.std = mean, std

        print("Computed mean:", self.mean, flush=True)
        print("Computed std:", self.std, flush=True)
        return mean, std

    
    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        paths = self.files[idx]
        source_dataset = self.source_dataset[idx]
        label = self.labels[idx]
        age_group = self.group_maps.get(source_dataset, "Adult")
        mean, std = self.mean[age_group], self.std[age_group]

        x_spec = []
        for path in paths:
            x_spec_i = np.load(path)
            x_spec_i = (x_spec_i - mean) / std  # Normalize
            x_spec.append(x_spec_i)
        x_spec = np.stack(x_spec, axis=0)


        # TODO make this dynamic
        num_total_channels = 6
        windows, n_channels, n_freq, n_times = x_spec.shape

        mask = np.zeros(num_total_channels, dtype=np.float32)
        if n_channels < num_total_channels:
            pad = np.zeros((windows, num_total_channels - n_channels, n_freq, n_times))
            x_spec = np.concatenate([x_spec, pad], axis=1) # Pad along channel dimension
            mask[:n_channels] = 1.0
        else:
            mask[:] = 1.0


        x = torch.tensor(x_spec, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)

        if self.is_train:
            channel_indices = torch.randperm(num_total_channels)
            x = x[:, channel_indices, :, :]
            mask = mask[channel_indices]

        return x, y, mask, self.source_dataset[idx]


class CnnEmbed(nn.Module):
    """Split EEG into fixed-size patches and embed them."""
    def __init__(self, emb_size=64, num_channels=6, height=50, width=122, dropout_prob=0.5):
        super().__init__()
        print(f"Initializing CnnEmbed with emb_size={emb_size}, num_channels={num_channels}, height={height}, width={width}, dropout_prob={dropout_prob}", flush=True)
        self.emb_size = emb_size

        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, emb_size, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, emb_size),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, height, width)
            cnn_out = self.cnn_feature_extractor(dummy)
            flattened_dim = cnn_out.view(1, -1).size(1)

        self.projection = nn.Linear(flattened_dim, emb_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        cnn_out = self.cnn_feature_extractor(x)
        flattened = cnn_out.view(cnn_out.size(0), -1)
        flattened = self.dropout(flattened)
        token = self.projection(flattened)

        return token

# -----------------------------
# Transformer-based model
# -----------------------------
class EEGTransformer(nn.Module):
    def __init__(self, emb_size=64, nhead=4, num_layers=4, num_classes=5, num_channels=6, height=50, width=122, dropout_prob=0.65):
        super().__init__()
        print(f"Initializing EEGTransformer with emb_size={emb_size}, nhead={nhead}, num_layers={num_layers}, num_classes={num_classes}, num_channels={num_channels}, height={height}, width={width}, dropout_prob={dropout_prob}", flush=True)

        self.emb_size = emb_size

        self.cnn_embed = CnnEmbed(
            emb_size=emb_size,
            num_channels=num_channels,
            height=height,
            dropout_prob=dropout_prob,
            width=width
        )

        self.pos_emb = nn.Embedding(10, emb_size)

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(emb_size, 1)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, segments, masks):
        # segments: (batch, 6, 15360)
        batch_size, window_size, num_channels, height, width = segments.shape

        # 1. Patch embedding
        segments_reshaped = segments.view(batch_size * window_size, num_channels, height, width)
        tokens = self.cnn_embed(segments_reshaped)
        tokens = tokens.view(batch_size, window_size, self.emb_size)

        # 2. Add positional embedding
        positions = torch.arange(window_size, device=segments.device)
        pos_emb = self.pos_emb(positions).unsqueeze(0)  # (1, window_size, emb_size)
        tokens = tokens + pos_emb

        # 3. Transformer
        # TODO DODAJ PADDING MASK , src_key_padding_mask=~masks
        transformer_out = self.transformer(tokens)  # (batch, 768, emb_size)

        # 4. Mean + learned pool across tokens
        weights = torch.softmax(self.attn_pool(transformer_out), dim=1)
        pooled_mean = transformer_out.mean(dim=1)
        pooled_attn = (transformer_out * weights).sum(dim=1)
        pooled = 0.5 * pooled_mean + 0.5 * pooled_attn

        # 5. Classify
        logits = self.classification_head(pooled)
        return logits


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
        for segments, labels, masks, source_datasets in data_loader:
            segments = segments.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            outputs = model(segments, masks)
            _, predicted = torch.max(outputs.data, 1)
            
            predicted_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(predicted_np)
            all_labels.extend(labels_np)

            # Store predictions and labels for each source
            for i in range(len(source_datasets)):
                source = source_datasets[i]
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


# --- Hyperparameters and Configuration ---
DATA_DIR = "../Data/"
TRAIN_PATIENTS = [
'PROCESSED_CAP_2/brux2',
'PROCESSED_CAP_2/ins1', 'PROCESSED_CAP_2/ins2', 'PROCESSED_CAP_2/ins3', 'PROCESSED_CAP_2/ins4', 'PROCESSED_CAP_2/ins5', 'PROCESSED_CAP_2/ins6',
'PROCESSED_CAP_2/n1', 'PROCESSED_CAP_2/n2', 'PROCESSED_CAP_2/n5', 'PROCESSED_CAP_2/n10',
'PROCESSED_CAP_2/narco1', 'PROCESSED_CAP_2/narco3',
'PROCESSED_CAP_2/nfle1', 'PROCESSED_CAP_2/nfle2', 'PROCESSED_CAP_2/nfle3', 'PROCESSED_CAP_2/nfle4', 'PROCESSED_CAP_2/nfle5', 'PROCESSED_CAP_2/nfle6', 'PROCESSED_CAP_2/nfle7', 'PROCESSED_CAP_2/nfle8', 'PROCESSED_CAP_2/nfle9', 'PROCESSED_CAP_2/nfle10', 'PROCESSED_CAP_2/nfle11', 'PROCESSED_CAP_2/nfle12', 'PROCESSED_CAP_2/nfle13', 'PROCESSED_CAP_2/nfle14', 'PROCESSED_CAP_2/nfle15', 'PROCESSED_CAP_2/nfle16', 'PROCESSED_CAP_2/nfle17', 'PROCESSED_CAP_2/nfle18', 'PROCESSED_CAP_2/nfle19', 'PROCESSED_CAP_2/nfle20', 'PROCESSED_CAP_2/nfle21', 'PROCESSED_CAP_2/nfle22', 'PROCESSED_CAP_2/nfle23', 'PROCESSED_CAP_2/nfle24', 'PROCESSED_CAP_2/nfle25', 'PROCESSED_CAP_2/nfle28', 'PROCESSED_CAP_2/nfle29', 'PROCESSED_CAP_2/nfle30', 'PROCESSED_CAP_2/nfle31', 'PROCESSED_CAP_2/nfle32', 'PROCESSED_CAP_2/nfle33',
'PROCESSED_CAP_2/plm1', 'PROCESSED_CAP_2/plm2', 'PROCESSED_CAP_2/plm3', 'PROCESSED_CAP_2/plm4', 'PROCESSED_CAP_2/plm5', 'PROCESSED_CAP_2/plm6', 'PROCESSED_CAP_2/plm7',
'PROCESSED_CAP_2/rbd1', 'PROCESSED_CAP_2/rbd2', 'PROCESSED_CAP_2/rbd3', 'PROCESSED_CAP_2/rbd4', 'PROCESSED_CAP_2/rbd5', 'PROCESSED_CAP_2/rbd6', 'PROCESSED_CAP_2/rbd7', 'PROCESSED_CAP_2/rbd8', 'PROCESSED_CAP_2/rbd9', 'PROCESSED_CAP_2/rbd10', 'PROCESSED_CAP_2/rbd11', 'PROCESSED_CAP_2/rbd12', 'PROCESSED_CAP_2/rbd13', 'PROCESSED_CAP_2/rbd15', 'PROCESSED_CAP_2/rbd16', 'PROCESSED_CAP_2/rbd17',
'PROCESSED_CAP_2/sdb1', 'PROCESSED_CAP_2/sdb2', 'PROCESSED_CAP_2/sdb3',
'PROCESSED_AHD/EPCTL01', 'PROCESSED_AHD/EPCTL02', 'PROCESSED_AHD/EPCTL03', 'PROCESSED_AHD/EPCTL04', 'PROCESSED_AHD/EPCTL05', 'PROCESSED_AHD/EPCTL06', 'PROCESSED_AHD/EPCTL07', 'PROCESSED_AHD/EPCTL08', 'PROCESSED_AHD/EPCTL09', 'PROCESSED_AHD/EPCTL10', 'PROCESSED_AHD/EPCTL11', 'PROCESSED_AHD/EPCTL12', 'PROCESSED_AHD/EPCTL13', 'PROCESSED_AHD/EPCTL14', 'PROCESSED_AHD/EPCTL15', 'PROCESSED_AHD/EPCTL16', 'PROCESSED_AHD/EPCTL17', 'PROCESSED_AHD/EPCTL18', 'PROCESSED_AHD/EPCTL19', 'PROCESSED_AHD/EPCTL20', 'PROCESSED_AHD/EPCTL21', 'PROCESSED_AHD/EPCTL22', 'PROCESSED_AHD/EPCTL23',
'PROCESSED_HFO/01', 'PROCESSED_HFO/02', 'PROCESSED_HFO/03', 'PROCESSED_HFO/04', 'PROCESSED_HFO/05', 'PROCESSED_HFO/06', 'PROCESSED_HFO/07', 'PROCESSED_HFO/08', 'PROCESSED_HFO/09', 'PROCESSED_HFO/11', 'PROCESSED_HFO/12', 'PROCESSED_HFO/13', 'PROCESSED_HFO/14', 'PROCESSED_HFO/15', 'PROCESSED_HFO/16', 'PROCESSED_HFO/17', 'PROCESSED_HFO/18', 'PROCESSED_HFO/19', 'PROCESSED_HFO/20', 'PROCESSED_HFO/21',
]

VALIDATION_PATIENTS = [
'PROCESSED_CAP_2/ins7', 'PROCESSED_CAP_2/ins8',
'PROCESSED_CAP_2/n3',
'PROCESSED_CAP_2/narco4',
'PROCESSED_CAP_2/nfle26', 'PROCESSED_CAP_2/nfle34', 'PROCESSED_CAP_2/nfle35', 'PROCESSED_CAP_2/nfle36', 'PROCESSED_CAP_2/nfle37',
'PROCESSED_CAP_2/plm8', 'PROCESSED_CAP_2/plm9',
'PROCESSED_CAP_2/rbd18', 'PROCESSED_CAP_2/rbd19', 'PROCESSED_CAP_2/rbd20',
'PROCESSED_CAP_2/sdb4',
'PROCESSED_AHD/EPCTL24', 'PROCESSED_AHD/EPCTL25', 'PROCESSED_AHD/EPCTL26',
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
LEARNING_RATE = 0.000015
WEIGHT_DECAY = 1e-3
EPOCHS = 25
EMB_SIZE = 32
NUM_CLASSES = 5
print(f"Using EMB_SIZE={EMB_SIZE}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}, EPOCHS={EPOCHS}", flush=True)

all_patients = sorted(list(set(TRAIN_PATIENTS + VALIDATION_PATIENTS + TEST_PATIENTS)))

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

#criterion = FocalLoss(task_type='multi-class', alpha=[0.5, 3.0, 1.0, 1.0, 1.0], gamma=2.0, reduction='mean', num_classes=NUM_CLASSES)
criterion = FocalLoss(task_type='multi-class', alpha=None, gamma=2.0, reduction='mean', num_classes=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# --- Training Loop ---
print("Starting training...")

# Check if the data directory and training data exist

PATIENCE = 8
current_patience = 0
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
    for segments, labels, masks, _ in train_loader:
        segments, labels, masks = segments.to(device), labels.to(device), masks.to(device)

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
        for segments, labels, masks, _ in validation_loader:
            segments, labels, masks = segments.to(device), labels.to(device), masks.to(device)
            outputs = model(segments, masks)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(validation_loader)
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print(f"Scheduler stepping with val_loss={avg_val_loss:.4f}, best so far={best_val_loss:.4f}", flush=True)
    
    scheduler.step()

    # Show current LR
    for pg in optimizer.param_groups:
        print(f"Current LR: {pg['lr']}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        current_patience = 0
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Validation loss decreased. Saving new best model.", flush=True)

    else:
        current_patience += 1
        if current_patience >= PATIENCE:
            print("Early stopping triggered.", flush=True)
            break

print("Training finished.")

# --- Evaluation on Test Set with the Best Model ---
print("\nLoading the best model for final evaluation on the test set...")
model.load_state_dict(torch.load('best_model.pth'))

if len(test_dataset) > 0:
    evaluate_and_print_cm(model, test_loader, device, ['W', 'N1', 'N2', 'N3', 'R'])
else:
    print("Warning: No test data found. Skipping final evaluation.")