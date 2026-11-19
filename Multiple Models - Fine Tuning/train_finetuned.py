import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mne
import glob
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
from focal_loss import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import argparse
import random

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EEGDataset(Dataset):
    def __init__(self, patients, data_dir, patient_to_id, filter_age_group, is_train=False, num_channels=6):
        self.is_train = is_train
        self.files = []
        self.labels = []
        self.patient_ids = []
        self.patient_to_id = patient_to_id
        self.filter_age_group = filter_age_group
        self.num_channels = num_channels

        self.LABEL_MAP = {
            "W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 3, "R": 4,
                    "N1": 1, "N2": 2, "N3": 3
        }
        self.AGE_MAP = {
            # 0 - Adult, 1 - Teen, 2 - Child, 3 - Infant
            "PROCESSED_CAP": 0, # Adult
            "PROCESSED_AHD": 0,
            "PROCESSED_HAAGLANDEN": 0,
            "PROCESSED_SLEEP_EDF": 0,
            "PROCESSED_NCH": -1, # TBD
        }
        self.age_df = pd.read_csv(os.path.join(data_dir,"PROCESSED_NCH","SLEEP_STUDY.csv"))
        self.age_df['FILENAME'] = self.age_df['STUDY_PAT_ID'].astype(str) + '_' + self.age_df['SLEEP_STUDY_ID'].astype(str)

        self.class_names = list(self.LABEL_MAP.keys())
        self.files_per_window = 10

        for patient_id_str in patients:
            folder = os.path.join(data_dir, f"{patient_id_str}")
            if not os.path.isdir(folder):
                print(f"Warning: Directory {folder} does not exist. Skipping.")
                continue
            
            npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")), 
                               key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            dataset_name = patient_id_str.split('/')[0] if '/' in patient_id_str else patient_id_str
            folder_name = os.path.basename(patient_id_str)
            patient_age_group = self.get_age_group(dataset_name, folder_name)

            if patient_age_group != self.filter_age_group:
                continue 

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
                else:
                    warnings.warn(f"Unknown label '{label_str}' in file {label_path}. Skipping this file.")
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

        print(f"Loaded {len(self.files)} samples for age group {self.filter_age_group} from {len(patients)} patients.", flush=True)

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

    def get_age_group(self, dataset_name, filename):
        age_group = self.AGE_MAP.get(dataset_name, 0) # Default to 0 if unknown
        
        if age_group == -1:
            row = self.age_df[self.age_df["FILENAME"] == filename]
            if row.empty:
                warnings.warn(f"Could not find age for {filename}. Defaulting to Adult.")
                return 0
            else:
                days = row.iloc[0]["AGE_AT_SLEEP_STUDY_DAYS"]
                return self.age_group_for_sleep(days)


        return age_group

    def age_group_for_sleep(self, days):
        if days < 2 * 365:
            return 3    # infant
        elif days < 12 * 365:
            return 2    # child
        elif days < 18 * 365:
            return 1    # teen
        else:
            return 0    # adult


    def __getitem__(self, idx):
        paths = self.files[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        dataset = paths[0].split('/')[-3]
        filename = paths[0].split('/')[-2]
        age_group = self.get_age_group(dataset, filename)

        if age_group == -1:
            filename = paths[0].split('/')[-2]  # Extract ID

            row = self.age_df[self.age_df["FILENAME"] == filename]

            if row.empty:
                warnings.warn(f"Could not find age for {filename}. Defaulting to Adult.")
                age_group = 0
            else:
                days = row.iloc[0]["AGE_AT_SLEEP_STUDY_DAYS"]
                age_group = self.age_group_for_sleep(days)

        x_spec = []
        for path in paths:
            x_spec_i = np.load(path)
            x_spec_i = (x_spec_i - self.mean) / self.std  # Normalize
            x_spec.append(x_spec_i)
        x_spec = np.stack(x_spec, axis=0)


        windows, n_channels, n_freq, n_times = x_spec.shape

        # 1. If data has MORE channels than requested, cut them
        if n_channels > self.num_channels:
            x_spec = x_spec[:, :self.num_channels, :, :]
            n_channels = self.num_channels

        mask = np.zeros(self.num_channels, dtype=np.float32)
        if n_channels < self.num_channels:
            pad = np.zeros((windows, self.num_channels - n_channels, n_freq, n_times))
            x_spec = np.concatenate([x_spec, pad], axis=1) # Pad along channel dimension
            mask[:n_channels] = 1.0
        else:
            mask[:] = 1.0


        x = torch.tensor(x_spec, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)

        if self.is_train:
            channel_indices = torch.randperm(self.num_channels)
            x = x[:, channel_indices, :, :]
            mask = mask[channel_indices]

        return dataset, x, y, patient_id, mask, age_group

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
    def __init__(self, emb_size=64, nhead=4, num_layers=3, num_classes=5, num_channels=6, height=50, width=122, dropout_prob=0.4):
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

    def forward(self, segments, patient_ids, masks):
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
        transformer_out = self.transformer(tokens)  # (batch, 768, emb_size)

        # 4. Mean + learned pool across tokens
        weights = torch.softmax(self.attn_pool(transformer_out), dim=1)
        pooled_mean = transformer_out.mean(dim=1)
        pooled_attn = (transformer_out * weights).sum(dim=1)
        pooled = 0.5 * pooled_mean + 0.5 * pooled_attn

        # 5. Classify
        logits = self.classification_head(pooled)
        return logits


# --- Evaluation Function ---
def evaluate_and_print_cm(model, data_loader, device, class_names, num_channels=None):
    """Evaluates the model and prints the confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    per_age_group_preds = {}
    per_age_group_labels = {}
    
    with torch.no_grad():
        for datasets, segments, labels, patient_ids, masks, age_groups in data_loader:
            segments = segments.to(device)
            labels = labels.to(device)
            patient_ids = patient_ids.to(device)
            masks = masks.to(device)
            age_groups = age_groups.to(device)
            segments_mod = segments.clone()
            if num_channels is not None:
                segments_mod[:, :, num_channels:, :, :] = 0.0

            outputs = model(segments_mod, patient_ids, masks)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i in range(len(age_groups)):
                age_group = int(age_groups[i])
                per_age_group_preds.setdefault(age_group, []).append(predicted.cpu().numpy()[i])
                per_age_group_labels.setdefault(age_group, []).append(labels.cpu().numpy()[i])

    for age_group in sorted(per_age_group_preds):
        ag_cm = confusion_matrix(per_age_group_labels[age_group], per_age_group_preds[age_group])
        accuracy = np.trace(ag_cm) / np.sum(ag_cm)
        print(f"Confusion Matrix for Age Group {age_group}:")
        print(ag_cm)
        print(f"Age Group: {age_group}, Accuracy: {accuracy:.4f}", flush=True)
        print("Cohen's Kappa:", cohen_kappa_score(per_age_group_labels[age_group], per_age_group_preds[age_group]), flush=True)
        print("F1 Score (macro):", f1_score(per_age_group_labels[age_group], per_age_group_preds[age_group], average='macro'), flush=True)
        print()
        print()

    # Compute the overall confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Evaluation Set (Patients 21-30)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Accuracy:", np.trace(cm) / np.sum(cm), flush=True)
    print("Cohen's Kappa:", cohen_kappa_score(all_labels, all_preds), flush=True)
    print("F1 Score (macro):", f1_score(all_labels, all_preds, average='macro'), flush=True)

if __name__ == "__main__":
    # --- Hyperparameters and Configuration ---
    DATA_DIR = "../../Data"
    TRAIN_PATIENTS = [
    'AASM/PROCESSED_CAP/n1', 'AASM/PROCESSED_CAP/n2', 'AASM/PROCESSED_CAP/n5', 'AASM/PROCESSED_CAP/n10',
    'AASM/PROCESSED_AHD/EPCTL01', 'AASM/PROCESSED_AHD/EPCTL02', 'AASM/PROCESSED_AHD/EPCTL03', 'AASM/PROCESSED_AHD/EPCTL04', 'AASM/PROCESSED_AHD/EPCTL05', 'AASM/PROCESSED_AHD/EPCTL06', 'AASM/PROCESSED_AHD/EPCTL07', 'AASM/PROCESSED_AHD/EPCTL08', 'AASM/PROCESSED_AHD/EPCTL09', 'AASM/PROCESSED_AHD/EPCTL10', 'AASM/PROCESSED_AHD/EPCTL11', 'AASM/PROCESSED_AHD/EPCTL12', 'AASM/PROCESSED_AHD/EPCTL13', 'AASM/PROCESSED_AHD/EPCTL14', 'AASM/PROCESSED_AHD/EPCTL15', 'AASM/PROCESSED_AHD/EPCTL16', 'AASM/PROCESSED_AHD/EPCTL17', 'AASM/PROCESSED_AHD/EPCTL18', 'AASM/PROCESSED_AHD/EPCTL19', 'AASM/PROCESSED_AHD/EPCTL20', 'AASM/PROCESSED_AHD/EPCTL21', 'AASM/PROCESSED_AHD/EPCTL22', 'AASM/PROCESSED_AHD/EPCTL23',
    'PROCESSED_HAAGLANDEN/SN001', 'PROCESSED_HAAGLANDEN/SN002', 'PROCESSED_HAAGLANDEN/SN003', 'PROCESSED_HAAGLANDEN/SN004', 'PROCESSED_HAAGLANDEN/SN005', 'PROCESSED_HAAGLANDEN/SN006', 'PROCESSED_HAAGLANDEN/SN007', 'PROCESSED_HAAGLANDEN/SN008', 'PROCESSED_HAAGLANDEN/SN009', 'PROCESSED_HAAGLANDEN/SN010', 'PROCESSED_HAAGLANDEN/SN011', 'PROCESSED_HAAGLANDEN/SN012', 'PROCESSED_HAAGLANDEN/SN013', 'PROCESSED_HAAGLANDEN/SN015', 'PROCESSED_HAAGLANDEN/SN016', 'PROCESSED_HAAGLANDEN/SN017', 'PROCESSED_HAAGLANDEN/SN018', 'PROCESSED_HAAGLANDEN/SN019', 'PROCESSED_HAAGLANDEN/SN020', 'PROCESSED_HAAGLANDEN/SN021', 'PROCESSED_HAAGLANDEN/SN022', 'PROCESSED_HAAGLANDEN/SN023', 'PROCESSED_HAAGLANDEN/SN024', 'PROCESSED_HAAGLANDEN/SN025', 'PROCESSED_HAAGLANDEN/SN026', 'PROCESSED_HAAGLANDEN/SN027', 'PROCESSED_HAAGLANDEN/SN028', 'PROCESSED_HAAGLANDEN/SN029', 'PROCESSED_HAAGLANDEN/SN030', 'PROCESSED_HAAGLANDEN/SN031', 'PROCESSED_HAAGLANDEN/SN032', 'PROCESSED_HAAGLANDEN/SN033', 'PROCESSED_HAAGLANDEN/SN034', 'PROCESSED_HAAGLANDEN/SN035', 'PROCESSED_HAAGLANDEN/SN036', 'PROCESSED_HAAGLANDEN/SN037', 'PROCESSED_HAAGLANDEN/SN038', 'PROCESSED_HAAGLANDEN/SN039', 'PROCESSED_HAAGLANDEN/SN040', 'PROCESSED_HAAGLANDEN/SN041', 'PROCESSED_HAAGLANDEN/SN042', 'PROCESSED_HAAGLANDEN/SN043', 'PROCESSED_HAAGLANDEN/SN044', 'PROCESSED_HAAGLANDEN/SN045', 'PROCESSED_HAAGLANDEN/SN046', 'PROCESSED_HAAGLANDEN/SN047', 'PROCESSED_HAAGLANDEN/SN048', 'PROCESSED_HAAGLANDEN/SN049', 'PROCESSED_HAAGLANDEN/SN050', 'PROCESSED_HAAGLANDEN/SN051', 'PROCESSED_HAAGLANDEN/SN052', 'PROCESSED_HAAGLANDEN/SN053', 'PROCESSED_HAAGLANDEN/SN054', 'PROCESSED_HAAGLANDEN/SN055', 'PROCESSED_HAAGLANDEN/SN056', 'PROCESSED_HAAGLANDEN/SN057', 'PROCESSED_HAAGLANDEN/SN058', 'PROCESSED_HAAGLANDEN/SN059', 'PROCESSED_HAAGLANDEN/SN060', 'PROCESSED_HAAGLANDEN/SN061', 'PROCESSED_HAAGLANDEN/SN062', 'PROCESSED_HAAGLANDEN/SN063', 'PROCESSED_HAAGLANDEN/SN065', 'PROCESSED_HAAGLANDEN/SN066', 'PROCESSED_HAAGLANDEN/SN067', 'PROCESSED_HAAGLANDEN/SN068', 'PROCESSED_HAAGLANDEN/SN069', 'PROCESSED_HAAGLANDEN/SN070', 'PROCESSED_HAAGLANDEN/SN071', 'PROCESSED_HAAGLANDEN/SN072', 'PROCESSED_HAAGLANDEN/SN073', 'PROCESSED_HAAGLANDEN/SN074', 'PROCESSED_HAAGLANDEN/SN075', 'PROCESSED_HAAGLANDEN/SN076', 'PROCESSED_HAAGLANDEN/SN077', 'PROCESSED_HAAGLANDEN/SN078', 'PROCESSED_HAAGLANDEN/SN079', 'PROCESSED_HAAGLANDEN/SN080', 'PROCESSED_HAAGLANDEN/SN081', 'PROCESSED_HAAGLANDEN/SN082', 'PROCESSED_HAAGLANDEN/SN083', 'PROCESSED_HAAGLANDEN/SN084', 'PROCESSED_HAAGLANDEN/SN085', 'PROCESSED_HAAGLANDEN/SN086', 'PROCESSED_HAAGLANDEN/SN087', 'PROCESSED_HAAGLANDEN/SN088', 'PROCESSED_HAAGLANDEN/SN089', 'PROCESSED_HAAGLANDEN/SN090', 'PROCESSED_HAAGLANDEN/SN091', 'PROCESSED_HAAGLANDEN/SN092', 'PROCESSED_HAAGLANDEN/SN093', 'PROCESSED_HAAGLANDEN/SN094', 'PROCESSED_HAAGLANDEN/SN095', 'PROCESSED_HAAGLANDEN/SN096', 'PROCESSED_HAAGLANDEN/SN097', 'PROCESSED_HAAGLANDEN/SN098', 'PROCESSED_HAAGLANDEN/SN099', 'PROCESSED_HAAGLANDEN/SN100', 'PROCESSED_HAAGLANDEN/SN101', 'PROCESSED_HAAGLANDEN/SN102', 'PROCESSED_HAAGLANDEN/SN103', 'PROCESSED_HAAGLANDEN/SN104', 'PROCESSED_HAAGLANDEN/SN105', 'PROCESSED_HAAGLANDEN/SN106', 'PROCESSED_HAAGLANDEN/SN107', 'PROCESSED_HAAGLANDEN/SN108', 'PROCESSED_HAAGLANDEN/SN109', 'PROCESSED_HAAGLANDEN/SN110', 'PROCESSED_HAAGLANDEN/SN111', 'PROCESSED_HAAGLANDEN/SN112', 'PROCESSED_HAAGLANDEN/SN113', 'PROCESSED_HAAGLANDEN/SN114', 'PROCESSED_HAAGLANDEN/SN115', 'PROCESSED_HAAGLANDEN/SN116', 'PROCESSED_HAAGLANDEN/SN117', 'PROCESSED_HAAGLANDEN/SN118', 'PROCESSED_HAAGLANDEN/SN119', 'PROCESSED_HAAGLANDEN/SN120', 'PROCESSED_HAAGLANDEN/SN121', 'PROCESSED_HAAGLANDEN/SN122', 'PROCESSED_HAAGLANDEN/SN123', 'PROCESSED_HAAGLANDEN/SN124',
    'PROCESSED_SLEEP_EDF/ST7011J0-PSG', 'PROCESSED_SLEEP_EDF/ST7012J0-PSG', 'PROCESSED_SLEEP_EDF/ST7021J0-PSG', 'PROCESSED_SLEEP_EDF/ST7022J0-PSG', 'PROCESSED_SLEEP_EDF/ST7041J0-PSG', 'PROCESSED_SLEEP_EDF/ST7042J0-PSG', 'PROCESSED_SLEEP_EDF/ST7051J0-PSG', 'PROCESSED_SLEEP_EDF/ST7052J0-PSG', 'PROCESSED_SLEEP_EDF/ST7061J0-PSG', 'PROCESSED_SLEEP_EDF/ST7062J0-PSG', 'PROCESSED_SLEEP_EDF/ST7071J0-PSG', 'PROCESSED_SLEEP_EDF/ST7072J0-PSG', 'PROCESSED_SLEEP_EDF/ST7081J0-PSG', 'PROCESSED_SLEEP_EDF/ST7082J0-PSG', 'PROCESSED_SLEEP_EDF/ST7091J0-PSG', 'PROCESSED_SLEEP_EDF/ST7092J0-PSG', 'PROCESSED_SLEEP_EDF/ST7101J0-PSG', 'PROCESSED_SLEEP_EDF/ST7102J0-PSG', 'PROCESSED_SLEEP_EDF/ST7111J0-PSG', 'PROCESSED_SLEEP_EDF/ST7112J0-PSG', 'PROCESSED_SLEEP_EDF/ST7121J0-PSG', 'PROCESSED_SLEEP_EDF/ST7122J0-PSG', 'PROCESSED_SLEEP_EDF/ST7131J0-PSG', 'PROCESSED_SLEEP_EDF/ST7132J0-PSG', 'PROCESSED_SLEEP_EDF/ST7141J0-PSG', 'PROCESSED_SLEEP_EDF/ST7142J0-PSG', 'PROCESSED_SLEEP_EDF/ST7151J0-PSG', 'PROCESSED_SLEEP_EDF/ST7152J0-PSG', 'PROCESSED_SLEEP_EDF/ST7161J0-PSG', 'PROCESSED_SLEEP_EDF/ST7162J0-PSG', 'PROCESSED_SLEEP_EDF/ST7171J0-PSG', 'PROCESSED_SLEEP_EDF/ST7172J0-PSG', 'PROCESSED_SLEEP_EDF/ST7181J0-PSG', 'PROCESSED_SLEEP_EDF/ST7182J0-PSG', 'PROCESSED_SLEEP_EDF/ST7191J0-PSG',
    'PROCESSED_NCH/10000_17728', 'PROCESSED_NCH/11017_21778', 'PROCESSED_NCH/12349_13747', 'PROCESSED_NCH/13591_12106', 'PROCESSED_NCH/15100_21748', 'PROCESSED_NCH/16957_15007', 'PROCESSED_NCH/19171_12841', 'PROCESSED_NCH/20695_24037', 'PROCESSED_NCH/3757_217', 'PROCESSED_NCH/5803_5752', 'PROCESSED_NCH/7570_1858', 'PROCESSED_NCH/9295_10033', 'PROCESSED_NCH/10135_9082', 'PROCESSED_NCH/11101_23188', 'PROCESSED_NCH/12448_12664', 'PROCESSED_NCH/136_4501', 'PROCESSED_NCH/154_18469', 'PROCESSED_NCH/17899_13756', 'PROCESSED_NCH/19171_8695', 'PROCESSED_NCH/20719_21973', 'PROCESSED_NCH/3805_19858', 'PROCESSED_NCH/6601_13297', 'PROCESSED_NCH/7612_25576', 'PROCESSED_NCH/9481_10345', 'PROCESSED_NCH/10297_14161', 'PROCESSED_NCH/11554_25465', 'PROCESSED_NCH/12448_21766', 'PROCESSED_NCH/13717_14743', 'PROCESSED_NCH/15541_26155', 'PROCESSED_NCH/17956_12826', 'PROCESSED_NCH/19690_18040', 'PROCESSED_NCH/20728_4291', 'PROCESSED_NCH/3838_14545', 'PROCESSED_NCH/6634_13186', 'PROCESSED_NCH/7639_5431', 'PROCESSED_NCH/10297_20458', 'PROCESSED_NCH/11575_3391', 'PROCESSED_NCH/127_19126', 'PROCESSED_NCH/14041_9991', 'PROCESSED_NCH/15925_4873', 'PROCESSED_NCH/18559_22930', 'PROCESSED_NCH/20305_17782', 'PROCESSED_NCH/2155_8443', 'PROCESSED_NCH/4051_9934', 'PROCESSED_NCH/6748_23851', 'PROCESSED_NCH/7735_2164', 'PROCESSED_NCH/10384_18907', 'PROCESSED_NCH/11650_24580', 'PROCESSED_NCH/13057_14236', 'PROCESSED_NCH/14368_15940', 'PROCESSED_NCH/15985_17278', 'PROCESSED_NCH/18850_9295', 'PROCESSED_NCH/20305_18697', 'PROCESSED_NCH/2704_18973', 'PROCESSED_NCH/4396_4000', 'PROCESSED_NCH/6943_17323', 'PROCESSED_NCH/7888_20158', 'PROCESSED_NCH/10768_298', 'PROCESSED_NCH/11800_17395', 'PROCESSED_NCH/13084_16093', 'PROCESSED_NCH/14401_18583', "PROCESSED_NCH/20281_15700","PROCESSED_NCH/3370_11521","PROCESSED_NCH/6484_9406", "PROCESSED_NCH/2155_9226","PROCESSED_NCH/4777_21505","PROCESSED_NCH/17638_20662", "PROCESSED_NCH/8011_19402","PROCESSED_NCH/8389_20047","PROCESSED_NCH/8986_18388", "PROCESSED_NCH/5209_16609","PROCESSED_NCH/5383_1240","PROCESSED_NCH/19447_16972", "PROCESSED_NCH/12379_1636","PROCESSED_NCH/6559_14521","PROCESSED_NCH/4954_9088", "PROCESSED_NCH/20695_18826","PROCESSED_NCH/11977_17332","PROCESSED_NCH/16225_10300", "PROCESSED_NCH/7003_9922","PROCESSED_NCH/3844_22312","PROCESSED_NCH/5461_16423", "PROCESSED_NCH/8689_3307","PROCESSED_NCH/1378_24625","PROCESSED_NCH/14725_1699", "PROCESSED_NCH/4849_8353","PROCESSED_NCH/91_3787","PROCESSED_NCH/9730_13141", "PROCESSED_NCH/17962_3343","PROCESSED_NCH/11980_6088","PROCESSED_NCH/20101_20149", "PROCESSED_NCH/15766_12988","PROCESSED_NCH/2404_12244","PROCESSED_NCH/20443_4693", "PROCESSED_NCH/18268_14539","PROCESSED_NCH/7699_16249","PROCESSED_NCH/7576_16669", "PROCESSED_NCH/13939_11092","PROCESSED_NCH/6712_19381", "PROCESSED_NCH/2569_2281", "PROCESSED_NCH/11689_3517", "PROCESSED_NCH/19660_7258", "PROCESSED_NCH/6334_3979", "PROCESSED_NCH/1843_3334", "PROCESSED_NCH/2461_21199", "PROCESSED_NCH/17188_25918", "PROCESSED_NCH/13585_11473", "PROCESSED_NCH/7372_8521", "PROCESSED_NCH/5065_4975", "PROCESSED_NCH/2587_15265", "PROCESSED_NCH/4642_2233", "PROCESSED_NCH/7591_22051", "PROCESSED_NCH/5488_20575", "PROCESSED_NCH/14146_14857", "PROCESSED_NCH/2428_7330", "PROCESSED_NCH/3547_19036", "PROCESSED_NCH/15808_11788", "PROCESSED_NCH/8062_6196", "PROCESSED_NCH/11227_17311", "PROCESSED_NCH/3865_4270", "PROCESSED_NCH/2038_15019", "PROCESSED_NCH/19789_2773", "PROCESSED_NCH/11686_6364", "PROCESSED_NCH/17860_11671", "PROCESSED_NCH/1408_16276", "PROCESSED_NCH/18169_2716", "PROCESSED_NCH/4099_19936", "PROCESSED_NCH/850_14323", "PROCESSED_NCH/16003_18829", "PROCESSED_NCH/1480_17857", "PROCESSED_NCH/9415_2623", "PROCESSED_NCH/17821_10558", "PROCESSED_NCH/18298_13927", "PROCESSED_NCH/20083_1624", "PROCESSED_NCH/9700_12472", "PROCESSED_NCH/5395_16471", "PROCESSED_NCH/8869_15730", "PROCESSED_NCH/16150_13177", "PROCESSED_NCH/5881_15205", "PROCESSED_NCH/6292_6409", "PROCESSED_NCH/187_6160", "PROCESSED_NCH/3349_23821", "PROCESSED_NCH/6850_22876", "PROCESSED_NCH/8959_17155", "PROCESSED_NCH/64_14908", "PROCESSED_NCH/13651_11308", "PROCESSED_NCH/8869_13885", "PROCESSED_NCH/4615_12328", "PROCESSED_NCH/13216_24934", "PROCESSED_NCH/100_23083", "PROCESSED_NCH/19033_8725", "PROCESSED_NCH/3607_10015", "PROCESSED_NCH/7387_4534", "PROCESSED_NCH/292_24565", "PROCESSED_NCH/3322_18940", "PROCESSED_NCH/9808_20281", "PROCESSED_NCH/20461_3574", "PROCESSED_NCH/12265_21259", "PROCESSED_NCH/15574_18004", "PROCESSED_NCH/1249_17515", "PROCESSED_NCH/16975_7444", "PROCESSED_NCH/2638_17656"

    ]

    VALIDATION_PATIENTS = [
    'AASM/PROCESSED_CAP/n3',
    'AASM/PROCESSED_AHD/EPCTL24', 'AASM/PROCESSED_AHD/EPCTL25', 'AASM/PROCESSED_AHD/EPCTL26',
    'PROCESSED_HAAGLANDEN/SN125', 'PROCESSED_HAAGLANDEN/SN126', 'PROCESSED_HAAGLANDEN/SN127', 'PROCESSED_HAAGLANDEN/SN128', 'PROCESSED_HAAGLANDEN/SN129', 'PROCESSED_HAAGLANDEN/SN130', 'PROCESSED_HAAGLANDEN/SN131', 'PROCESSED_HAAGLANDEN/SN132', 'PROCESSED_HAAGLANDEN/SN133', 'PROCESSED_HAAGLANDEN/SN134', 'PROCESSED_HAAGLANDEN/SN136', 'PROCESSED_HAAGLANDEN/SN137', 'PROCESSED_HAAGLANDEN/SN138', 'PROCESSED_HAAGLANDEN/SN139', 'PROCESSED_HAAGLANDEN/SN140',
    'PROCESSED_SLEEP_EDF/ST7192J0-PSG', 'PROCESSED_SLEEP_EDF/ST7201J0-PSG', 'PROCESSED_SLEEP_EDF/ST7202J0-PSG', 'PROCESSED_SLEEP_EDF/ST7211J0-PSG',
    'PROCESSED_NCH/16063_19705', 'PROCESSED_NCH/18916_23833', 'PROCESSED_NCH/20329_14086', 'PROCESSED_NCH/2884_9004', 'PROCESSED_NCH/4771_7681', 'PROCESSED_NCH/7081_19480', 'PROCESSED_NCH/8431_21844', 'PROCESSED_NCH/1084_17239', 'PROCESSED_NCH/1180_11506', 'PROCESSED_NCH/13279_20710', 'PROCESSED_NCH/14587_26149', 'PROCESSED_NCH/1630_25783', 'PROCESSED_NCH/18949_4846', 'PROCESSED_NCH/20353_24562', 'PROCESSED_NCH/3220_994', 'PROCESSED_NCH/4840_4558', 'PROCESSED_NCH/718_5569', 'PROCESSED_NCH/8581_13543', "PROCESSED_NCH/16999_11215","PROCESSED_NCH/11971_26134","PROCESSED_NCH/13882_11578", "PROCESSED_NCH/4147_24535","PROCESSED_NCH/9730_17305","PROCESSED_NCH/7024_4144", "PROCESSED_NCH/19171_3772","PROCESSED_NCH/910_13882", "PROCESSED_NCH/10288_7846", "PROCESSED_NCH/17176_4636", "PROCESSED_NCH/1189_6484", "PROCESSED_NCH/6235_16354", "PROCESSED_NCH/12754_22234", "PROCESSED_NCH/1726_8539", "PROCESSED_NCH/20071_24805", "PROCESSED_NCH/6880_21946", "PROCESSED_NCH/15478_17371", "PROCESSED_NCH/14506_1057", "PROCESSED_NCH/4447_12001", "PROCESSED_NCH/5416_3730", "PROCESSED_NCH/1687_16795", "PROCESSED_NCH/17497_10585",
    ]

    TEST_PATIENTS = [
    'AASM/PROCESSED_CAP/n11',
    'AASM/PROCESSED_AHD/EPCTL27', 'AASM/PROCESSED_AHD/EPCTL28', 'AASM/PROCESSED_AHD/EPCTL29',
    'PROCESSED_HAAGLANDEN/SN141', 'PROCESSED_HAAGLANDEN/SN142', 'PROCESSED_HAAGLANDEN/SN143', 'PROCESSED_HAAGLANDEN/SN144', 'PROCESSED_HAAGLANDEN/SN145', 'PROCESSED_HAAGLANDEN/SN146', 'PROCESSED_HAAGLANDEN/SN147', 'PROCESSED_HAAGLANDEN/SN148', 'PROCESSED_HAAGLANDEN/SN149', 'PROCESSED_HAAGLANDEN/SN150', 'PROCESSED_HAAGLANDEN/SN151', 'PROCESSED_HAAGLANDEN/SN152', 'PROCESSED_HAAGLANDEN/SN153', 'PROCESSED_HAAGLANDEN/SN154',
    'PROCESSED_SLEEP_EDF/ST7212J0-PSG', 'PROCESSED_SLEEP_EDF/ST7221J0-PSG', 'PROCESSED_SLEEP_EDF/ST7222J0-PSG', 'PROCESSED_SLEEP_EDF/ST7241J0-PSG', 'PROCESSED_SLEEP_EDF/ST7242J0-PSG',
    'PROCESSED_NCH/10855_25105', 'PROCESSED_NCH/12058_5692', 'PROCESSED_NCH/13393_10039', 'PROCESSED_NCH/145_1537', 'PROCESSED_NCH/16609_10486', 'PROCESSED_NCH/19108_12343', 'PROCESSED_NCH/20443_2872', 'PROCESSED_NCH/3442_24379', 'PROCESSED_NCH/484_24694', 'PROCESSED_NCH/7276_9796', 'PROCESSED_NCH/8668_6322', 'PROCESSED_NCH/10_22339', 'PROCESSED_NCH/1216_23932', 'PROCESSED_NCH/13576_22957', 'PROCESSED_NCH/14869_23599', 'PROCESSED_NCH/16765_7066', 'PROCESSED_NCH/19156_21742', 'PROCESSED_NCH/20644_17932', 'PROCESSED_NCH/3481_1288', 'PROCESSED_NCH/5491_17983', 'PROCESSED_NCH/7465_9982', 'PROCESSED_NCH/8992_13411', "PROCESSED_NCH/3481_5146","PROCESSED_NCH/20032_3643","PROCESSED_NCH/4141_19918", "PROCESSED_NCH/19975_23428","PROCESSED_NCH/7888_4660","PROCESSED_NCH/4228_6760", "PROCESSED_NCH/19648_24769","PROCESSED_NCH/8281_7117","PROCESSED_NCH/8707_23248", "PROCESSED_NCH/13432_7204","PROCESSED_NCH/964_10150","PROCESSED_NCH/16342_12895", "PROCESSED_NCH/17440_6772","PROCESSED_NCH/14623_8203", "PROCESSED_NCH/13615_40", "PROCESSED_NCH/5779_21415", "PROCESSED_NCH/6664_26107", "PROCESSED_NCH/15634_8032", "PROCESSED_NCH/2716_1702", "PROCESSED_NCH/20386_1846", "PROCESSED_NCH/17437_22696", "PROCESSED_NCH/12265_25651", "PROCESSED_NCH/19843_20713", "PROCESSED_NCH/12919_11878", "PROCESSED_NCH/6256_20272", "PROCESSED_NCH/5119_21151", "PROCESSED_NCH/5617_4393"
    ]

    parser = argparse.ArgumentParser(description='EEG Training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_channels', type=int, default=6, help='Number of EEG channels to use')
    parser.add_argument('--age_group', type=int, default=None, help='Age group to filter: 0=Adult, 1=Teen, 2=Child, 3=Infant.')
    args = parser.parse_args()

    EMB_SIZE = args.emb_size
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs
    DROPOUT_PROB = args.dropout
    NUM_LAYERS = args.num_layers
    NHEAD = args.nhead
    AGE_GROUP = args.age_group
    PATIENCE = 5
    NUM_CLASSES = 5
    NUM_CHANNELS = args.num_channels

    model_name = f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random.randint(1, 100000)}.pth"
    print(f"Model will be saved to: {model_name}", flush=True)

    print(f"Using EMB_SIZE={EMB_SIZE}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}, EPOCHS={EPOCHS}, NUM_CHANNELS={NUM_CHANNELS}", flush=True)

    all_patients = sorted(list(set(TRAIN_PATIENTS + VALIDATION_PATIENTS + TEST_PATIENTS)))
    PATIENT_TO_ID = {patient_name: i for i, patient_name in enumerate(all_patients)}

    # --- Dataset and DataLoader ---
    train_dataset = EEGDataset(
        patients=TRAIN_PATIENTS,
        data_dir=DATA_DIR,
        patient_to_id=PATIENT_TO_ID,
        filter_age_group=AGE_GROUP,
        is_train=True,
        num_channels=NUM_CHANNELS
    )
    train_mean, train_std = train_dataset.get_mean_std()

    age_counts = Counter(d[5] for d in train_dataset)
    print("Training data age group distribution:", age_counts)

    train_age_groups = [d[5] for d in train_dataset]
    class_counts = np.bincount(train_age_groups)
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[age] for age in train_age_groups])
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        shuffle=False
    )

    validation_dataset = EEGDataset(
        patients=VALIDATION_PATIENTS,
        data_dir=DATA_DIR,
        patient_to_id=PATIENT_TO_ID,
        filter_age_group=AGE_GROUP,
        num_channels=NUM_CHANNELS
    )
    validation_dataset.set_mean_std(train_mean, train_std)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False) # No shuffle for validation

    height, width = train_dataset[0][1].shape[2], train_dataset[0][1].shape[3]
    print(f"Input spectrogram dimensions: Height={height}, Width={width}", flush=True)

    test_dataset = EEGDataset(
        patients=TEST_PATIENTS,
        data_dir=DATA_DIR,
        patient_to_id=PATIENT_TO_ID,
        filter_age_group=AGE_GROUP,
        num_channels=NUM_CHANNELS
    )
    test_dataset.set_mean_std(train_mean, train_std)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False) # No shuffle for evaluation

    # --- Model, Loss, and Optimizer ---
    model = EEGTransformer(
        emb_size=EMB_SIZE,
        num_classes=NUM_CLASSES,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        num_channels=NUM_CHANNELS,
        height=height,
        width=width,
        dropout_prob=DROPOUT_PROB
    ).to(device)

    model_path = "adult_only_model.pth" if NUM_CHANNELS == 6 else "adult_only_model_3ch.pth"
    print(f"Loading pre-trained model from: {model_path}", flush=True)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer.parameters():
        param.requires_grad = True

    for param in model.classification_head.parameters():
        param.requires_grad = True


    criterion = FocalLoss(task_type='multi-class', alpha=None, gamma=2.0, reduction='mean', num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # --- Training Loop ---
    print("Starting training...")

    # Check if the data directory and training data exist

    current_patience = 0
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for _, segments, labels, patient_ids, masks, age_group in train_loader:
            segments, labels, patient_ids, masks, age_group = segments.to(device), labels.to(device), patient_ids.to(device), masks.to(device), age_group.to(device)

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
            for _, segments, labels, patient_ids, masks, age_group in validation_loader:
                segments, labels, patient_ids, masks, age_group = segments.to(device), labels.to(device), patient_ids.to(device), masks.to(device), age_group.to(device)
                outputs = model(segments, patient_ids, masks)
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
            torch.save(model.state_dict(), model_name)
            print(f"Validation loss decreased. Saving new best model.", flush=True)

        else:
            current_patience += 1
            if current_patience >= PATIENCE:
                print("Early stopping triggered.", flush=True)
                break

    print("Training finished.")

    # --- Evaluation on Test Set with the Best Model ---
    print("\nLoading the best model for final evaluation on the test set...")
    model.load_state_dict(torch.load(model_name))

    if len(test_dataset) > 0:
        print("\nEvaluating on the test set...", flush=True)
        evaluate_and_print_cm(model, test_loader, device, ['W', 'N1', 'N2', 'N3', 'R'])
        print("\nEvaluating on the test set with only the first 3 channels...", flush=True)
        evaluate_and_print_cm(model, test_loader, device, ['W', 'N1', 'N2', 'N3', 'R'], num_channels=3)
    else:
        print("Warning: No test data found. Skipping final evaluation.")