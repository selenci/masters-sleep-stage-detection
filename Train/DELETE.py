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
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EEGDataset(Dataset):
    def __init__(self, patients, data_dir, patient_to_id):
        self.files = []
        self.labels = []
        self.patient_ids = []
        self.patient_to_id = patient_to_id

        self.LABEL_MAP = {
            "W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 3, "R": 4,
        }
        self.class_names = list(self.LABEL_MAP.keys())
        self.files_per_window = 10

        for patient_id_str in patients:
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
                    self.patient_ids.append(self.patient_to_id[patient_id_str])
                    self.labels.append(center_label)

    def __len__(self):
        return len(self.files)

    def get_mean_std(self):
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        unique_files = set([item for sublist in self.files for item in sublist])
        for path in unique_files:
            x_spec = np.load(path)
            x = x_spec.ravel()  # flatten, no copy
            total_sum += x.sum()
            total_sq_sum += np.square(x).sum()
            total_count += x.size

        mean = total_sum / total_count
        # E[x^2] - (E[x])^2
        var = (total_sq_sum / total_count) - mean**2
        std = np.sqrt(var)

        self.mean, self.std = mean, std

        print("Computed mean:", self.mean, flush=True)
        print("Computed std:", self.std, flush=True)
        return mean, std

    
    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        paths = self.files[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        x_spec = []
        for path in paths:
            x_spec_i = np.load(path)
            x_spec_i = (x_spec_i - self.mean) / self.std  # Normalize
            x_spec.append(x_spec_i)
        x_spec = np.stack(x_spec, axis=0)

        windows, n_channels, n_freq, n_times = x_spec.shape

        mask = np.zeros(6, dtype=np.float32)
        if n_channels < 6:
            pad = np.zeros((windows, 6 - n_channels, n_freq, n_times))
            x_spec = np.concatenate([x_spec, pad], axis=1) # Pad along channel dimension
            mask[:n_channels] = 1.0
        else:
            mask[:] = 1.0


        x = torch.tensor(x_spec, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)

        return x, y, patient_id, mask


class CnnEmbed(nn.Module):
    """Split EEG into fixed-size patches and embed them."""
    def __init__(self, emb_size=64, num_channels=6, height=50, width=122, dropout_prob=0.5):
        super().__init__()
        self.emb_size = emb_size

        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, emb_size, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(emb_size),
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
    def __init__(self, emb_size=64, nhead=4, num_layers=3, num_classes=5, max_patients=30, num_channels=6, height=50, width=122, dropout_prob=0.5):
        super().__init__()
        self.emb_size = emb_size

        self.cnn_embed = CnnEmbed(
            emb_size=emb_size,
            num_channels=num_channels,
            height=height,
            width=width
        )

        self.patient_emb = nn.Embedding(max_patients + 1, emb_size)

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Classification head
        self.classification_head = nn.Linear(emb_size, num_classes)

    def forward(self, segments, patient_ids, masks):
        # segments: (batch, 6, 15360)
        batch_size, window_size, num_channels, height, width = segments.shape

        # 1. Patch embedding
        segments_reshaped = segments.view(batch_size * window_size, num_channels, height, width)
        tokens = self.cnn_embed(segments_reshaped)
        tokens = tokens.view(batch_size, window_size, self.emb_size)
        # 2. Add patient embedding (broadcast to all tokens)
        patient_emb = self.patient_emb(patient_ids).unsqueeze(1)  # (batch ,emb_size)
        tokens = tokens + patient_emb

        # 3. Transformer
        # TODO DODAJ PADDING MASK , src_key_padding_mask=~masks
        transformer_out = self.transformer(tokens)  # (batch, 768, emb_size)

        # 4. Mean pool across tokens
        pooled = transformer_out.mean(dim=1)  # (batch, emb_size)
        pooled = self.dropout(pooled)

        # 5. Classify
        logits = self.classification_head(pooled)
        return logits

# --- Evaluation Function ---
def evaluate_and_print_cm(model, data_loader, device, class_names):
    """Evaluates the model and prints the confusion matrix."""
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for segments, labels, patient_ids, masks in data_loader:
            segments = segments.to(device)
            labels = labels.to(device)
            patient_ids = patient_ids.to(device)
            masks = masks.to(device)

            outputs = model(segments, patient_ids, masks)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Evaluation Set (Patients 21-30)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')

    print("Accuracy:", np.trace(cm) / np.sum(cm), flush=True)


# --- Hyperparameters and Configuration ---
DATA_DIR = "../Data/PROCESSED_CAP"
TRAIN_PATIENTS = [
    'brux2',
    'ins1', 'ins2', 'ins3', 'ins4', 'ins5',
    'n1', 'n2', 'n5', 'n10',
    'narco1', 'narco3',
    'nfle1', 'nfle2', 'nfle3', 'nfle4', 'nfle5', 'nfle7', 'nfle8',
    'nfle12', 'nfle13', 'nfle14', 'nfle15', 'nfle16', 'nfle17',
    'nfle21', 'nfle22', 'nfle28', 'nfle29', 'nfle32',
    'plm1', 'plm2', 'plm3', 'plm4', 'plm5',
    'rbd1', 'rbd2', 'rbd3', 'rbd4', 'rbd5', 'rbd6', 'rbd7', 'rbd8',
    'rbd10', 'rbd11', 'rbd12', 'rbd15', 'rbd16',
    'sdb1', 'sdb2',
]

VALIDATION_PATIENTS = [
    'ins6', 'ins7',
    'n3',
    'narco4',
    'nfle9', 'nfle18', 'nfle24', 'nfle30', 'nfle34',
    'plm6', 'plm7', 'plm8',
    'rbd9', 'rbd13', 'rbd17', 'rbd18',
    'sdb3',
]

TEST_PATIENTS = [
    'ins8', 'ins9',
    'n11',
    'narco5',
    'nfle35', 'nfle36', 'nfle37', 'nfle38', 'nfle39', 'nfle40',
    'plm9', 'plm10',
    'rbd19', 'rbd20', 'rbd21', 'rbd22',
    'sdb4',
]


BATCH_SIZE = 16
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EMB_SIZE = 64
NUM_CLASSES = 5

all_patients = sorted(list(set(TRAIN_PATIENTS + VALIDATION_PATIENTS + TEST_PATIENTS)))
PATIENT_TO_ID = {patient_name: i for i, patient_name in enumerate(all_patients)}
MAX_PATIENTS = len(all_patients)

# --- Dataset and DataLoader ---
train_dataset = EEGDataset(
    patients=TRAIN_PATIENTS,
    data_dir=DATA_DIR,
    patient_to_id=PATIENT_TO_ID
)
train_mean, train_std = train_dataset.get_mean_std()

validation_dataset = EEGDataset(
    patients=VALIDATION_PATIENTS,
    data_dir=DATA_DIR,
    patient_to_id=PATIENT_TO_ID
)
validation_dataset.set_mean_std(train_mean, train_std)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for validation

height, width = train_dataset[0][0].shape[2], train_dataset[0][0].shape[3]
print(f"Input spectrogram dimensions: Height={height}, Width={width}", flush=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = EEGDataset(
    patients=TEST_PATIENTS,
    data_dir=DATA_DIR,
    patient_to_id=PATIENT_TO_ID
)
test_dataset.set_mean_std(train_mean, train_std)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for evaluation

# --- Model, Loss, and Optimizer ---
model = EEGTransformer(
    emb_size=EMB_SIZE,
    num_classes=NUM_CLASSES,
    max_patients=MAX_PATIENTS,
    num_channels=6,
    height=height,
    width=width
).to(device)

criterion = FocalLoss(task_type='multi-class', alpha=None, gamma=2.0, reduction='mean', num_classes=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# --- Training Loop ---
print("Starting training...")

# Check if the data directory and training data exist

PATIENCE = 10
current_patience = 0
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
    for segments, labels, patient_ids, masks in train_loader:
        segments, labels, patient_ids, masks = segments.to(device), labels.to(device), patient_ids.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(segments, patient_ids, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for segments, labels, patient_ids, masks in validation_loader:
            segments, labels, patient_ids, masks = segments.to(device), labels.to(device), patient_ids.to(device), masks.to(device)
            outputs = model(segments, patient_ids, masks)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(validation_loader)
    scheduler.step(avg_val_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}", flush=True)

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