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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


model_name = 'best_model' + "_3" + '.pth' 
print(f"Model will be saved to: {model_name}", flush=True)

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EEGDataset(Dataset):
    def __init__(self, patients, data_dir, patient_to_id, is_train=False):
        self.is_train = is_train
        self.files = []
        self.labels = []
        self.patient_ids = []
        self.patient_to_id = patient_to_id

        self.LABEL_MAP = {
            "W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 3, "R": 4,
                    "N1": 1, "N2": 2, "N3": 3
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

        dataset = paths[0].split('/')[2]

        x_spec = []
        for path in paths:
            x_spec_i = np.load(path)
            x_spec_i = (x_spec_i - self.mean) / self.std  # Normalize
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

        return dataset, x, y, patient_id, mask


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
    def __init__(self, emb_size=64, nhead=4, num_layers=3, num_classes=5, num_channels=6, height=50, width=122, dropout_prob=0.5):
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

# --- Evaluation Function ---
def evaluate_and_print_cm(model, data_loader, device, class_names):
    """Evaluates the model and prints the confusion matrix."""
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    per_dataset_preds = {}
    per_dataset_labels = {}
    
    with torch.no_grad():
        for datasets, segments, labels, patient_ids, masks in data_loader:
            segments = segments.to(device)
            labels = labels.to(device)
            patient_ids = patient_ids.to(device)
            masks = masks.to(device)

            outputs = model(segments, patient_ids, masks)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(datasets)):
                dataset = datasets[i]
                per_dataset_preds.setdefault(dataset, []).append(predicted.cpu().numpy()[i])
                per_dataset_labels.setdefault(dataset, []).append(labels.cpu().numpy()[i])

    for dataset in per_dataset_preds:
        dataset_cm = confusion_matrix(per_dataset_labels[dataset], per_dataset_preds[dataset])
        accuracy = np.trace(dataset_cm) / np.sum(dataset_cm)
        print(f"Confusion Matrix for {dataset}:")
        print(dataset_cm)

        print(f"Dataset: {dataset}, Accuracy: {accuracy:.4f}", flush=True)
        print("Cohen's Kappa:", cohen_kappa_score(per_dataset_labels[dataset], per_dataset_preds[dataset]) , flush=True)
        print("F1 Score (macro):", f1_score(per_dataset_labels[dataset], per_dataset_preds[dataset], average='macro'), flush=True)
        print()
        print()

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


    print("Cohen's Kappa:", cohen_kappa_score(all_labels, all_preds), flush=True)
    print("F1 Score (macro):", f1_score(all_labels, all_preds, average='macro'), flush=True)


# --- Hyperparameters and Configuration ---
DATA_DIR = "../Data"
TRAIN_PATIENTS = [
'PROCESSED_CAP_3/brux2',
'PROCESSED_CAP_3/ins1', 'PROCESSED_CAP_3/ins2', 'PROCESSED_CAP_3/ins3', 'PROCESSED_CAP_3/ins4', 'PROCESSED_CAP_3/ins5', 'PROCESSED_CAP_3/ins6',
'PROCESSED_CAP_3/n1', 'PROCESSED_CAP_3/n2', 'PROCESSED_CAP_3/n5', 'PROCESSED_CAP_3/n10',
'PROCESSED_CAP_3/narco1', 'PROCESSED_CAP_3/narco3',
'PROCESSED_CAP_3/nfle1', 'PROCESSED_CAP_3/nfle2', 'PROCESSED_CAP_3/nfle3', 'PROCESSED_CAP_3/nfle4', 'PROCESSED_CAP_3/nfle5', 'PROCESSED_CAP_3/nfle6', 'PROCESSED_CAP_3/nfle7', 'PROCESSED_CAP_3/nfle8', 'PROCESSED_CAP_3/nfle9', 'PROCESSED_CAP_3/nfle10', 'PROCESSED_CAP_3/nfle11', 'PROCESSED_CAP_3/nfle12', 'PROCESSED_CAP_3/nfle13', 'PROCESSED_CAP_3/nfle14', 'PROCESSED_CAP_3/nfle15', 'PROCESSED_CAP_3/nfle16', 'PROCESSED_CAP_3/nfle17', 'PROCESSED_CAP_3/nfle18', 'PROCESSED_CAP_3/nfle19', 'PROCESSED_CAP_3/nfle20', 'PROCESSED_CAP_3/nfle21', 'PROCESSED_CAP_3/nfle22', 'PROCESSED_CAP_3/nfle23', 'PROCESSED_CAP_3/nfle24', 'PROCESSED_CAP_3/nfle25', 'PROCESSED_CAP_3/nfle28', 'PROCESSED_CAP_3/nfle29', 'PROCESSED_CAP_3/nfle30', 'PROCESSED_CAP_3/nfle31', 'PROCESSED_CAP_3/nfle32', 'PROCESSED_CAP_3/nfle33',
'PROCESSED_CAP_3/plm1', 'PROCESSED_CAP_3/plm2', 'PROCESSED_CAP_3/plm3', 'PROCESSED_CAP_3/plm4', 'PROCESSED_CAP_3/plm5', 'PROCESSED_CAP_3/plm6', 'PROCESSED_CAP_3/plm7',
'PROCESSED_CAP_3/rbd1', 'PROCESSED_CAP_3/rbd2', 'PROCESSED_CAP_3/rbd3', 'PROCESSED_CAP_3/rbd4', 'PROCESSED_CAP_3/rbd5', 'PROCESSED_CAP_3/rbd6', 'PROCESSED_CAP_3/rbd7', 'PROCESSED_CAP_3/rbd8', 'PROCESSED_CAP_3/rbd9', 'PROCESSED_CAP_3/rbd10', 'PROCESSED_CAP_3/rbd11', 'PROCESSED_CAP_3/rbd12', 'PROCESSED_CAP_3/rbd13', 'PROCESSED_CAP_3/rbd15', 'PROCESSED_CAP_3/rbd16', 'PROCESSED_CAP_3/rbd17',
'PROCESSED_CAP_3/sdb1', 'PROCESSED_CAP_3/sdb2', 'PROCESSED_CAP_3/sdb3',
'PROCESSED_AHD_2/EPCTL01', 'PROCESSED_AHD_2/EPCTL02', 'PROCESSED_AHD_2/EPCTL03', 'PROCESSED_AHD_2/EPCTL04', 'PROCESSED_AHD_2/EPCTL05', 'PROCESSED_AHD_2/EPCTL06', 'PROCESSED_AHD_2/EPCTL07', 'PROCESSED_AHD_2/EPCTL08', 'PROCESSED_AHD_2/EPCTL09', 'PROCESSED_AHD_2/EPCTL10', 'PROCESSED_AHD_2/EPCTL11', 'PROCESSED_AHD_2/EPCTL12', 'PROCESSED_AHD_2/EPCTL13', 'PROCESSED_AHD_2/EPCTL14', 'PROCESSED_AHD_2/EPCTL15', 'PROCESSED_AHD_2/EPCTL16', 'PROCESSED_AHD_2/EPCTL17', 'PROCESSED_AHD_2/EPCTL18', 'PROCESSED_AHD_2/EPCTL19', 'PROCESSED_AHD_2/EPCTL20', 'PROCESSED_AHD_2/EPCTL21', 'PROCESSED_AHD_2/EPCTL22', 'PROCESSED_AHD_2/EPCTL23',
'PROCESSED_HFO_2/01', 'PROCESSED_HFO_2/02', 'PROCESSED_HFO_2/03', 'PROCESSED_HFO_2/04', 'PROCESSED_HFO_2/05', 'PROCESSED_HFO_2/06', 'PROCESSED_HFO_2/07', 'PROCESSED_HFO_2/08', 'PROCESSED_HFO_2/09', 'PROCESSED_HFO_2/11', 'PROCESSED_HFO_2/12', 'PROCESSED_HFO_2/13', 'PROCESSED_HFO_2/14', 'PROCESSED_HFO_2/15', 'PROCESSED_HFO_2/16', 'PROCESSED_HFO_2/17', 'PROCESSED_HFO_2/18', 'PROCESSED_HFO_2/19', 'PROCESSED_HFO_2/20', 'PROCESSED_HFO_2/21',
'PROCESSED_HAAGLANDEN/SN001', 'PROCESSED_HAAGLANDEN/SN002', 'PROCESSED_HAAGLANDEN/SN003', 'PROCESSED_HAAGLANDEN/SN004', 'PROCESSED_HAAGLANDEN/SN005', 'PROCESSED_HAAGLANDEN/SN006', 'PROCESSED_HAAGLANDEN/SN007', 'PROCESSED_HAAGLANDEN/SN008', 'PROCESSED_HAAGLANDEN/SN009', 'PROCESSED_HAAGLANDEN/SN010', 'PROCESSED_HAAGLANDEN/SN011', 'PROCESSED_HAAGLANDEN/SN012', 'PROCESSED_HAAGLANDEN/SN013', 'PROCESSED_HAAGLANDEN/SN015', 'PROCESSED_HAAGLANDEN/SN016', 'PROCESSED_HAAGLANDEN/SN017', 'PROCESSED_HAAGLANDEN/SN018', 'PROCESSED_HAAGLANDEN/SN019', 'PROCESSED_HAAGLANDEN/SN020', 'PROCESSED_HAAGLANDEN/SN021', 'PROCESSED_HAAGLANDEN/SN022', 'PROCESSED_HAAGLANDEN/SN023', 'PROCESSED_HAAGLANDEN/SN024', 'PROCESSED_HAAGLANDEN/SN025', 'PROCESSED_HAAGLANDEN/SN026', 'PROCESSED_HAAGLANDEN/SN027', 'PROCESSED_HAAGLANDEN/SN028', 'PROCESSED_HAAGLANDEN/SN029', 'PROCESSED_HAAGLANDEN/SN030', 'PROCESSED_HAAGLANDEN/SN031', 'PROCESSED_HAAGLANDEN/SN032', 'PROCESSED_HAAGLANDEN/SN033', 'PROCESSED_HAAGLANDEN/SN034', 'PROCESSED_HAAGLANDEN/SN035', 'PROCESSED_HAAGLANDEN/SN036', 'PROCESSED_HAAGLANDEN/SN037', 'PROCESSED_HAAGLANDEN/SN038', 'PROCESSED_HAAGLANDEN/SN039', 'PROCESSED_HAAGLANDEN/SN040', 'PROCESSED_HAAGLANDEN/SN041', 'PROCESSED_HAAGLANDEN/SN042', 'PROCESSED_HAAGLANDEN/SN043', 'PROCESSED_HAAGLANDEN/SN044', 'PROCESSED_HAAGLANDEN/SN045', 'PROCESSED_HAAGLANDEN/SN046', 'PROCESSED_HAAGLANDEN/SN047', 'PROCESSED_HAAGLANDEN/SN048', 'PROCESSED_HAAGLANDEN/SN049', 'PROCESSED_HAAGLANDEN/SN050', 'PROCESSED_HAAGLANDEN/SN051', 'PROCESSED_HAAGLANDEN/SN052', 'PROCESSED_HAAGLANDEN/SN053', 'PROCESSED_HAAGLANDEN/SN054', 'PROCESSED_HAAGLANDEN/SN055', 'PROCESSED_HAAGLANDEN/SN056', 'PROCESSED_HAAGLANDEN/SN057', 'PROCESSED_HAAGLANDEN/SN058', 'PROCESSED_HAAGLANDEN/SN059', 'PROCESSED_HAAGLANDEN/SN060', 'PROCESSED_HAAGLANDEN/SN061', 'PROCESSED_HAAGLANDEN/SN062', 'PROCESSED_HAAGLANDEN/SN063', 'PROCESSED_HAAGLANDEN/SN065', 'PROCESSED_HAAGLANDEN/SN066', 'PROCESSED_HAAGLANDEN/SN067', 'PROCESSED_HAAGLANDEN/SN068', 'PROCESSED_HAAGLANDEN/SN069', 'PROCESSED_HAAGLANDEN/SN070', 'PROCESSED_HAAGLANDEN/SN071', 'PROCESSED_HAAGLANDEN/SN072', 'PROCESSED_HAAGLANDEN/SN073', 'PROCESSED_HAAGLANDEN/SN074', 'PROCESSED_HAAGLANDEN/SN075', 'PROCESSED_HAAGLANDEN/SN076', 'PROCESSED_HAAGLANDEN/SN077', 'PROCESSED_HAAGLANDEN/SN078', 'PROCESSED_HAAGLANDEN/SN079', 'PROCESSED_HAAGLANDEN/SN080', 'PROCESSED_HAAGLANDEN/SN081', 'PROCESSED_HAAGLANDEN/SN082', 'PROCESSED_HAAGLANDEN/SN083', 'PROCESSED_HAAGLANDEN/SN084', 'PROCESSED_HAAGLANDEN/SN085', 'PROCESSED_HAAGLANDEN/SN086', 'PROCESSED_HAAGLANDEN/SN087', 'PROCESSED_HAAGLANDEN/SN088', 'PROCESSED_HAAGLANDEN/SN089', 'PROCESSED_HAAGLANDEN/SN090', 'PROCESSED_HAAGLANDEN/SN091', 'PROCESSED_HAAGLANDEN/SN092', 'PROCESSED_HAAGLANDEN/SN093', 'PROCESSED_HAAGLANDEN/SN094', 'PROCESSED_HAAGLANDEN/SN095', 'PROCESSED_HAAGLANDEN/SN096', 'PROCESSED_HAAGLANDEN/SN097', 'PROCESSED_HAAGLANDEN/SN098', 'PROCESSED_HAAGLANDEN/SN099', 'PROCESSED_HAAGLANDEN/SN100', 'PROCESSED_HAAGLANDEN/SN101', 'PROCESSED_HAAGLANDEN/SN102', 'PROCESSED_HAAGLANDEN/SN103', 'PROCESSED_HAAGLANDEN/SN104', 'PROCESSED_HAAGLANDEN/SN105', 'PROCESSED_HAAGLANDEN/SN106', 'PROCESSED_HAAGLANDEN/SN107', 'PROCESSED_HAAGLANDEN/SN108', 'PROCESSED_HAAGLANDEN/SN109', 'PROCESSED_HAAGLANDEN/SN110', 'PROCESSED_HAAGLANDEN/SN111', 'PROCESSED_HAAGLANDEN/SN112', 'PROCESSED_HAAGLANDEN/SN113', 'PROCESSED_HAAGLANDEN/SN114', 'PROCESSED_HAAGLANDEN/SN115', 'PROCESSED_HAAGLANDEN/SN116', 'PROCESSED_HAAGLANDEN/SN117', 'PROCESSED_HAAGLANDEN/SN118', 'PROCESSED_HAAGLANDEN/SN119', 'PROCESSED_HAAGLANDEN/SN120', 'PROCESSED_HAAGLANDEN/SN121', 'PROCESSED_HAAGLANDEN/SN122', 'PROCESSED_HAAGLANDEN/SN123', 'PROCESSED_HAAGLANDEN/SN124',
'PROCESSED_SLEEP_EDF/ST7011J0-PSG', 'PROCESSED_SLEEP_EDF/ST7012J0-PSG', 'PROCESSED_SLEEP_EDF/ST7021J0-PSG', 'PROCESSED_SLEEP_EDF/ST7022J0-PSG', 'PROCESSED_SLEEP_EDF/ST7041J0-PSG', 'PROCESSED_SLEEP_EDF/ST7042J0-PSG', 'PROCESSED_SLEEP_EDF/ST7051J0-PSG', 'PROCESSED_SLEEP_EDF/ST7052J0-PSG', 'PROCESSED_SLEEP_EDF/ST7061J0-PSG', 'PROCESSED_SLEEP_EDF/ST7062J0-PSG', 'PROCESSED_SLEEP_EDF/ST7071J0-PSG', 'PROCESSED_SLEEP_EDF/ST7072J0-PSG', 'PROCESSED_SLEEP_EDF/ST7081J0-PSG', 'PROCESSED_SLEEP_EDF/ST7082J0-PSG', 'PROCESSED_SLEEP_EDF/ST7091J0-PSG', 'PROCESSED_SLEEP_EDF/ST7092J0-PSG', 'PROCESSED_SLEEP_EDF/ST7101J0-PSG', 'PROCESSED_SLEEP_EDF/ST7102J0-PSG', 'PROCESSED_SLEEP_EDF/ST7111J0-PSG', 'PROCESSED_SLEEP_EDF/ST7112J0-PSG', 'PROCESSED_SLEEP_EDF/ST7121J0-PSG', 'PROCESSED_SLEEP_EDF/ST7122J0-PSG', 'PROCESSED_SLEEP_EDF/ST7131J0-PSG', 'PROCESSED_SLEEP_EDF/ST7132J0-PSG', 'PROCESSED_SLEEP_EDF/ST7141J0-PSG', 'PROCESSED_SLEEP_EDF/ST7142J0-PSG', 'PROCESSED_SLEEP_EDF/ST7151J0-PSG', 'PROCESSED_SLEEP_EDF/ST7152J0-PSG', 'PROCESSED_SLEEP_EDF/ST7161J0-PSG', 'PROCESSED_SLEEP_EDF/ST7162J0-PSG', 'PROCESSED_SLEEP_EDF/ST7171J0-PSG', 'PROCESSED_SLEEP_EDF/ST7172J0-PSG', 'PROCESSED_SLEEP_EDF/ST7181J0-PSG', 'PROCESSED_SLEEP_EDF/ST7182J0-PSG', 'PROCESSED_SLEEP_EDF/ST7191J0-PSG',
'PROCESSED_NCH/10000_17728', 'PROCESSED_NCH/10855_25105', 'PROCESSED_NCH/10_22339', 'PROCESSED_NCH/11101_23188', 'PROCESSED_NCH/11554_25465', 'PROCESSED_NCH/1180_11506', 'PROCESSED_NCH/127_19126', 'PROCESSED_NCH/13084_16093', 'PROCESSED_NCH/13393_10039', 'PROCESSED_NCH/13717_14743', 'PROCESSED_NCH/14401_18583', 'PROCESSED_NCH/145_1537', 'PROCESSED_NCH/154_18469', 'PROCESSED_NCH/16063_19705', 'PROCESSED_NCH/17956_12826', 'PROCESSED_NCH/18916_23833', 'PROCESSED_NCH/19156_21742', 'PROCESSED_NCH/20329_14086', 'PROCESSED_NCH/20353_24562', 'PROCESSED_NCH/20644_17932', 'PROCESSED_NCH/20695_24037', 'PROCESSED_NCH/20719_21973', 'PROCESSED_NCH/20728_4291', 'PROCESSED_NCH/2155_8443', 'PROCESSED_NCH/2884_9004',
]

VALIDATION_PATIENTS = [
'PROCESSED_CAP_3/ins7', 'PROCESSED_CAP_3/ins8',
'PROCESSED_CAP_3/n3',
'PROCESSED_CAP_3/narco4',
'PROCESSED_CAP_3/nfle26', 'PROCESSED_CAP_3/nfle34', 'PROCESSED_CAP_3/nfle35', 'PROCESSED_CAP_3/nfle36', 'PROCESSED_CAP_3/nfle37',
'PROCESSED_CAP_3/plm8', 'PROCESSED_CAP_3/plm9',
'PROCESSED_CAP_3/rbd18', 'PROCESSED_CAP_3/rbd19', 'PROCESSED_CAP_3/rbd20',
'PROCESSED_CAP_3/sdb4',
'PROCESSED_AHD_2/EPCTL24', 'PROCESSED_AHD_2/EPCTL25', 'PROCESSED_AHD_2/EPCTL26',
'PROCESSED_HFO_2/22', 'PROCESSED_HFO_2/23', 'PROCESSED_HFO_2/24', 'PROCESSED_HFO_2/25', 'PROCESSED_HFO_2/26',
'PROCESSED_HAAGLANDEN/SN125', 'PROCESSED_HAAGLANDEN/SN126', 'PROCESSED_HAAGLANDEN/SN127', 'PROCESSED_HAAGLANDEN/SN128', 'PROCESSED_HAAGLANDEN/SN129', 'PROCESSED_HAAGLANDEN/SN130', 'PROCESSED_HAAGLANDEN/SN131', 'PROCESSED_HAAGLANDEN/SN132', 'PROCESSED_HAAGLANDEN/SN133', 'PROCESSED_HAAGLANDEN/SN134', 'PROCESSED_HAAGLANDEN/SN136', 'PROCESSED_HAAGLANDEN/SN137', 'PROCESSED_HAAGLANDEN/SN138', 'PROCESSED_HAAGLANDEN/SN139', 'PROCESSED_HAAGLANDEN/SN140',
'PROCESSED_SLEEP_EDF/ST7192J0-PSG', 'PROCESSED_SLEEP_EDF/ST7201J0-PSG', 'PROCESSED_SLEEP_EDF/ST7202J0-PSG', 'PROCESSED_SLEEP_EDF/ST7211J0-PSG',
'PROCESSED_NCH/3220_994', 'PROCESSED_NCH/3757_217', 'PROCESSED_NCH/7081_19480',
]

TEST_PATIENTS = [
'PROCESSED_CAP_3/ins9',
'PROCESSED_CAP_3/n11',
'PROCESSED_CAP_3/narco5',
'PROCESSED_CAP_3/nfle38', 'PROCESSED_CAP_3/nfle39', 'PROCESSED_CAP_3/nfle40',
'PROCESSED_CAP_3/plm10',
'PROCESSED_CAP_3/rbd21', 'PROCESSED_CAP_3/rbd22',
'PROCESSED_AHD_2/EPCTL27', 'PROCESSED_AHD_2/EPCTL28', 'PROCESSED_AHD_2/EPCTL29',
'PROCESSED_HFO_2/27', 'PROCESSED_HFO_2/28', 'PROCESSED_HFO_2/29', 'PROCESSED_HFO_2/30',
'PROCESSED_HAAGLANDEN/SN141', 'PROCESSED_HAAGLANDEN/SN142', 'PROCESSED_HAAGLANDEN/SN143', 'PROCESSED_HAAGLANDEN/SN144', 'PROCESSED_HAAGLANDEN/SN145', 'PROCESSED_HAAGLANDEN/SN146', 'PROCESSED_HAAGLANDEN/SN147', 'PROCESSED_HAAGLANDEN/SN148', 'PROCESSED_HAAGLANDEN/SN149', 'PROCESSED_HAAGLANDEN/SN150', 'PROCESSED_HAAGLANDEN/SN151', 'PROCESSED_HAAGLANDEN/SN152', 'PROCESSED_HAAGLANDEN/SN153', 'PROCESSED_HAAGLANDEN/SN154',
'PROCESSED_SLEEP_EDF/ST7212J0-PSG', 'PROCESSED_SLEEP_EDF/ST7221J0-PSG', 'PROCESSED_SLEEP_EDF/ST7222J0-PSG', 'PROCESSED_SLEEP_EDF/ST7241J0-PSG', 'PROCESSED_SLEEP_EDF/ST7242J0-PSG',
'PROCESSED_NCH/718_5569', 'PROCESSED_NCH/8431_21844', 'PROCESSED_NCH/8992_13411', 'PROCESSED_NCH/9295_10033',
]

BATCH_SIZE = 16
LEARNING_RATE = 0.000055
WEIGHT_DECAY = 1e-3
EPOCHS = 30
EMB_SIZE = 32
NUM_CLASSES = 5
print(f"Using EMB_SIZE={EMB_SIZE}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}, EPOCHS={EPOCHS}", flush=True)

all_patients = sorted(list(set(TRAIN_PATIENTS + VALIDATION_PATIENTS + TEST_PATIENTS)))
PATIENT_TO_ID = {patient_name: i for i, patient_name in enumerate(all_patients)}

# --- Dataset and DataLoader ---
train_dataset = EEGDataset(
    patients=TRAIN_PATIENTS,
    data_dir=DATA_DIR,
    patient_to_id=PATIENT_TO_ID,
    is_train=True
)
train_mean, train_std = train_dataset.get_mean_std()

validation_dataset = EEGDataset(
    patients=VALIDATION_PATIENTS,
    data_dir=DATA_DIR,
    patient_to_id=PATIENT_TO_ID
)
validation_dataset.set_mean_std(train_mean, train_std)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for validation

height, width = train_dataset[0][1].shape[2], train_dataset[0][1].shape[3]
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
    num_channels=6,
    height=height,
    width=width
).to(device)

criterion = FocalLoss(task_type='multi-class', alpha=None, gamma=2.0, reduction='mean', num_classes=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

# --- Training Loop ---
print("Starting training...")

# Check if the data directory and training data exist

PATIENCE = 5
current_patience = 0
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
    for _, segments, labels, patient_ids, masks in train_loader:
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
        for _, segments, labels, patient_ids, masks in validation_loader:
            segments, labels, patient_ids, masks = segments.to(device), labels.to(device), patient_ids.to(device), masks.to(device)
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
    evaluate_and_print_cm(model, test_loader, device, ['W', 'N1', 'N2', 'N3', 'R'])
else:
    print("Warning: No test data found. Skipping final evaluation.")