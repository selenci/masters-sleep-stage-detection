import argparse
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'R']
LABEL_MAP = {
    "W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 3, "R": 4,
    "N1": 1, "N2": 2, "N3": 3,
}

def load_module_from_file(module_name: str, filepath: str):
    path = Path(filepath).resolve()
    dir_path = str(path.parent)

    # 1. Prepend the model's directory so local imports (like focal_loss) resolve
    sys.path.insert(0, dir_path)

    # 2. Evict previous local collision modules from sys.modules cache
    sys.modules.pop("focal_loss", None)

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        # 3. Clean up sys.path so it doesn't pollute subsequent imports
        if dir_path in sys.path:
            sys.path.remove(dir_path)


def age_group_for_sleep(days):
    # Fall back to Baseline (0) if age_in_days is missing, None, NaN/Inf, negative, or non-numeric
    if days is None:
        return 0
    try:
        days = float(days)
    except (TypeError, ValueError):
        return 0

    if not math.isfinite(days) or days < 0:
        return 0

    if days < 2 * 365:
        return 3    # infant
    elif days < 12 * 365:
        return 2    # child
    elif days < 18 * 365:
        return 1    # teen
    else:
        return 0    # adult / baseline


# Cache loaded models by (age_group, variant_key) so each is loaded at most once
LOADED_MODELS = {}

MODEL_CONFIGS = {
    0: {  # adult / baseline
        "module_name": "baseline_model",
        "python_file": "../Baseline/train.py",
        "class_name": "EEGTransformer",
        "base_params": {
            "emb_size": 64,
            "nhead": 8,
            "num_layers": 2,
            "num_classes": 5,
            "height": 100,
            "width": 32,
            "dropout_prob": 0.4,
        },
        "variants": {
            "3ch": {
                "weights_path": "../Baseline/exp1_3ch_best.pth",
                "mean": 9.34156e-07,
                "std": 5.2653377e-06,
                "params": {"num_channels": 3},
            },
            "6ch": {
                "weights_path": "../Baseline/exp1_6ch_best.pth",
                "mean": 9.341711e-07,
                "std": 5.2653577e-06,
                "params": {"num_channels": 6},
            },
        },
    },
    1: {  # teen
        "module_name": "teen_model",
        "python_file": "../Multiple Models - Fine Tuning/train_finetuned.py",
        "class_name": "EEGTransformer",
        "base_params": {
            "emb_size": 64,
            "nhead": 8,
            "num_layers": 2,
            "num_classes": 5,
            "height": 100,
            "width": 32,
            "dropout_prob": 0.4,
        },
        "variants": {
            "3ch": {
                "weights_path": "../Multiple Models - Fine Tuning/exp4_3ch_teen.pth",
                "mean": 9.604172e-07,
                "std": 3.9469096e-06,
                "params": {"num_channels": 3},
            },
            "6ch": {
                "weights_path": "../Multiple Models - Fine Tuning/exp4_6ch_teen.pth",
                "mean": 9.604119e-07,
                "std": 3.9469073e-06,
                "params": {"num_channels": 6},
            },
        },
    },
    2: {  # child
        "module_name": "child_model",
        "python_file": "../One Model - Embeddings/train_normalize_per_group.py",
        "class_name": "EEGTransformer",
        "base_params": {
            "emb_size": 64,
            "nhead": 4,
            "num_classes": 5,
            "height": 100,
            "width": 32,
        },
        "variants": {
            "3ch": {
                "weights_path": "../One Model - Embeddings/exp2_3ch_best.pth",
                "mean": 1.5295395314751659e-06,
                "std": 7.863315659051295e-06,
                "params": {
                    "num_channels": 3,
                    "num_layers": 4,
                    "dropout_prob": 0.5,
                },
            },
            "6ch": {
                "weights_path": "../One Model - Embeddings/exp2_6ch_best.pth",
                "mean": 1.529541e-06,
                "std": 7.863311e-06,
                "params": {
                    "num_channels": 6,
                    "num_layers": 3,
                    "dropout_prob": 0.4,
                },
            },
        },
    },
    3: {  # infant
        "module_name": "infant_model",
        "python_file": "../Multiple Models - Seperate/train_multiple_models.py",
        "class_name": "EEGTransformer",
        "base_params": {
            "emb_size": 64,
            "nhead": 8,
            "num_layers": 2,
            "num_classes": 5,
            "height": 100,
            "width": 32,
            "dropout_prob": 0.4,
        },
        "variants": {
            "3ch": {
                "weights_path": "../Multiple Models - Seperate/exp3_3ch_babies.pth",
                "mean": 1.4594199e-06,
                "std": 8.2557635e-06,
                "params": {"num_channels": 3},
            },
            "6ch": {
                "weights_path": "../Multiple Models - Seperate/exp3_6ch_babies.pth",
                "mean": 1.4594253e-06,
                "std": 8.255783e-06,
                "params": {"num_channels": 6},
            },
        },
    },
}


def model_picker(age_in_days, input_channels, target_device):
    age_group = age_group_for_sleep(age_in_days)
    group_config = MODEL_CONFIGS[age_group]
    variants = group_config["variants"]

    variant_key = "3ch" if (input_channels <= 3 and "3ch" in variants) else "6ch"
    cache_key = (age_group, variant_key)

    if cache_key in LOADED_MODELS:
        return LOADED_MODELS[cache_key]

    variant = variants[variant_key]
    params = {**group_config["base_params"], **variant.get("params", {})}

    weights_path = variant["weights_path"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights file not found for age_group={age_group} ({variant_key}): {weights_path}"
        )

    mod = load_module_from_file(group_config["module_name"], group_config["python_file"])
    ModelClass = getattr(mod, group_config["class_name"])
    model = ModelClass(**params)

    state_dict = torch.load(weights_path, map_location=target_device, weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    LOADED_MODELS[cache_key] = {
        "model": model,
        "age_group": age_group,
        "variant": variant_key,
        "mean": variant["mean"],
        "std": variant["std"],
        "num_channels": params["num_channels"],
    }
    return LOADED_MODELS[cache_key]


def compute_patient_statistics(y_true, y_pred, total_predicted_epochs, unmapped_labels):
    """Compute accuracy, Cohen's kappa, F1 scores, per-stage metrics, and confusion matrix."""
    if not y_true:
        return None

    num_classes = len(STAGE_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)

    total_eval = int(cm.sum())
    correct = int(np.trace(cm))
    incorrect = total_eval - correct
    accuracy = correct / total_eval

    row_sums = cm.sum(axis=1)  # Ground truth support per class
    col_sums = cm.sum(axis=0)  # Predicted count per class

    expected_acc = float(np.sum(row_sums * col_sums)) / (total_eval ** 2)
    if (1.0 - expected_acc) > 0:
        cohens_kappa = (accuracy - expected_acc) / (1.0 - expected_acc)
    else:
        cohens_kappa = 0.0

    per_stage = {}
    present_f1s = []

    for idx, stage in enumerate(STAGE_NAMES):
        tp = int(cm[idx, idx])
        gt_count = int(row_sums[idx])
        pred_count = int(col_sums[idx])

        precision = (tp / pred_count) if pred_count > 0 else 0.0
        recall = (tp / gt_count) if gt_count > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if gt_count > 0:
            present_f1s.append(f1)

        per_stage[stage] = {
            "ground_truth_count": gt_count,
            "predicted_count": pred_count,
            "ground_truth_pct": round(100.0 * gt_count / total_eval, 2),
            "predicted_pct": round(100.0 * pred_count / total_eval, 2),
            "correct": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = float(np.mean(present_f1s)) if present_f1s else 0.0

    stats = {
        "total_predicted_epochs": total_predicted_epochs,
        "evaluated_epochs": total_eval,
        "unmapped_ground_truth_epochs": len(unmapped_labels),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(cohens_kappa, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": {
            "labels": STAGE_NAMES,
            "rows": "ground_truth",
            "columns": "predicted",
            "matrix": cm.tolist(),
        },
        "per_stage": per_stage,
    }

    if unmapped_labels:
        stats["unmapped_labels_seen"] = sorted(set(unmapped_labels))

    return stats


def print_patient_statistics(patient_id, variant, input_channels, stats):
    """Pretty-print evaluation metrics and confusion matrix to stdout."""
    print(
        f"  -> Results for {patient_id} ({variant}, {input_channels}ch): "
        f"{stats['correct']}/{stats['evaluated_epochs']} correct "
        f"({stats['accuracy'] * 100:.2f}%) | "
        f"Macro F1: {stats['macro_f1']:.4f} | "
        f"Kappa: {stats['cohens_kappa']:.4f}"
    )

    if stats["unmapped_ground_truth_epochs"] > 0:
        print(
            f"     Note: {stats['unmapped_ground_truth_epochs']} epoch(s) had unmapped "
            f"ground-truth labels: {stats.get('unmapped_labels_seen', [])}"
        )

    print("     Per-Stage Breakdown:")
    print("     Stage   GT    Pred  Right   Prec    Recall    F1     GT%     Pred%")
    for stage in STAGE_NAMES:
        s = stats["per_stage"][stage]
        print(
            f"     {stage:<5} {s['ground_truth_count']:>4}  {s['predicted_count']:>4}  "
            f"{s['correct']:>5}  {s['precision']:>6.4f}  {s['recall']:>6.4f}  {s['f1']:>6.4f}  "
            f"{s['ground_truth_pct']:>5.1f}%  {s['predicted_pct']:>5.1f}%"
        )

    cm = stats["confusion_matrix"]["matrix"]
    print("     Confusion Matrix (rows = GT, cols = Pred):")
    header = "           " + " ".join(f"{name:>5}" for name in STAGE_NAMES)
    print(header)
    for stage, row in zip(STAGE_NAMES, cm):
        row_str = " ".join(f"{val:>5}" for val in row)
        print(f"     {stage:<5} {row_str}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-model EEG sleep stage inference based on patient age and channel count.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSON File Format Help:
----------------------
The input JSON must contain a list of recording objects. Each object should include:
  - "folder_path": (str) Directory containing the .npy and optional -stage.txt files
  - "age_in_days": (int/float, optional) Age of the patient in days (uses Baseline model if missing/null/NaN)
  - "patient_id": (str/int, optional) Unique identifier

Example folder structure:
  Data/Dataset/Patient_ID/
    ├── 0000.npy
    ├── 0000-stage.txt  (optional ground-truth)
    ├── 0001.npy
    └── 0001-stage.txt

Example JSON file:
[
  {
    "patient_id": "12_345678",
    "age_in_days": 3650,
    "folder_path": "Data/Dataset/12_345678"
  }
]
        """
    )

    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the input JSON file containing recording metadata and folder paths."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("inference_results.json"),
        help="Path to save the output predictions JSON (default: inference_results.json)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: auto-detect 'cuda' or 'cpu')."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.json_file.is_file():
        sys.exit(f"Error: Specified file does not exist: {args.json_file}")

    try:
        with open(args.json_file, 'r') as f:
            recordings = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Could not parse JSON file '{args.json_file}': {e}")

    if not isinstance(recordings, list):
        sys.exit("Error: Expected JSON root to be a list of recording objects.")

    target_device = torch.device(args.device)
    print(f"Loaded {len(recordings)} recording(s). Using device: {target_device}")

    all_results = []
    evaluated_summaries = []
    window_size = 10

    for recording in recordings:
        patient_id = recording.get("patient_id", "unknown")
        age_in_days = recording.get("age_in_days")
        folder = recording.get("folder_path") or recording.get("folder")

        if not folder or not os.path.isdir(folder):
            print(f"Warning: Directory '{folder}' does not exist for patient {patient_id}. Skipping.")
            continue

        # Sort all .npy files numerically (e.g. 0000.npy, 0001.npy, ...)
        npy_files = sorted(
            glob.glob(os.path.join(folder, "*.npy")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        if len(npy_files) < window_size:
            print(f"Skipping {patient_id}: Found {len(npy_files)} files, but need at least {window_size}.")
            continue

        # Load raw spectrograms once so we can run both full and 3ch-truncated passes
        raw_specs = [np.load(f) for f in npy_files]
        raw_channels = raw_specs[0].shape[0]
        channel_runs = [raw_channels, 3] if raw_channels > 3 else [raw_channels]

        for input_channels in channel_runs:
            loaded = model_picker(age_in_days, input_channels, target_device)

            model = loaded["model"]
            age_group = loaded["age_group"]
            mean = loaded["mean"]
            std = loaded["std"]
            model_channels = loaded["num_channels"]

            print(
                f"Running inference for patient {patient_id} "
                f"(Age Group: {age_group}, Variant: {loaded['variant']}, "
                f"Input Channels: {input_channels}/{raw_channels}, Total Epochs: {len(npy_files)})..."
            )

            loaded_specs = []
            for x_raw in raw_specs:
                x_i = x_raw[:input_channels]
                if std > 0:
                    x_i = (x_i - mean) / std
                loaded_specs.append(x_i)

            patient_predictions = []
            y_true = []
            y_pred = []
            unmapped_labels = []

            with torch.no_grad():
                for i in range(len(npy_files) - window_size + 1):
                    window_specs = loaded_specs[i : i + window_size]
                    center_file = npy_files[i + (window_size // 2)]

                    # Stack window: (10, channels, freq, time)
                    x_spec = np.stack(window_specs, axis=0)

                    # Channel padding / clipping to the selected model's num_channels
                    cur_channels = x_spec.shape[1]
                    mask = np.zeros(model_channels, dtype=np.float32)

                    if cur_channels > model_channels:
                        x_spec = x_spec[:, :model_channels, :, :]
                        mask[:] = 1.0
                    elif cur_channels < model_channels:
                        pad = np.zeros(
                            (window_size, model_channels - cur_channels, x_spec.shape[2], x_spec.shape[3]),
                            dtype=np.float32,
                        )
                        x_spec = np.concatenate([x_spec, pad], axis=1)
                        mask[:cur_channels] = 1.0
                    else:
                        mask[:] = 1.0

                    segments = torch.tensor(x_spec, dtype=torch.float32).unsqueeze(0).to(target_device)
                    masks = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(target_device)
                    patient_ids = torch.tensor([0], dtype=torch.long).to(target_device)

                    if age_group == 2:
                        age_tensor = torch.tensor([age_group], dtype=torch.long).to(target_device)
                        outputs = model(segments, patient_ids, masks, age_tensor)
                    else:
                        outputs = model(segments, patient_ids, masks)

                    pred_idx = torch.argmax(outputs, dim=1).item()
                    pred_stage = STAGE_NAMES[pred_idx]

                    gt_stage_file = os.path.splitext(center_file)[0] + "-stage.txt"
                    ground_truth = None
                    if os.path.isfile(gt_stage_file):
                        with open(gt_stage_file, "r") as gf:
                            ground_truth = gf.read().strip()

                    record_info = {
                        "epoch_file": os.path.basename(center_file),
                        "predicted_stage": pred_stage,
                        "predicted_idx": pred_idx,
                    }
                    if ground_truth:
                        record_info["ground_truth"] = ground_truth
                        gt_idx = LABEL_MAP.get(ground_truth.upper())
                        if gt_idx is not None:
                            record_info["ground_truth_stage"] = STAGE_NAMES[gt_idx]
                            record_info["ground_truth_idx"] = gt_idx
                            record_info["correct"] = (pred_idx == gt_idx)
                            y_true.append(gt_idx)
                            y_pred.append(pred_idx)
                        else:
                            unmapped_labels.append(ground_truth)

                    patient_predictions.append(record_info)

            stats = compute_patient_statistics(
                y_true=y_true,
                y_pred=y_pred,
                total_predicted_epochs=len(patient_predictions),
                unmapped_labels=unmapped_labels,
            )

            result_entry = {
                "patient_id": patient_id,
                "age_group": age_group,
                "model_variant": loaded["variant"],
                "input_channels": input_channels,
                "folder": folder,
                "predictions": patient_predictions,
            }
            if stats is not None:
                result_entry["statistics"] = stats
                print_patient_statistics(patient_id, loaded["variant"], input_channels, stats)
                evaluated_summaries.append({
                    "patient_id": str(patient_id),
                    "variant": loaded["variant"],
                    "channels": input_channels,
                    "correct": stats["correct"],
                    "total": stats["evaluated_epochs"],
                    "accuracy": stats["accuracy"],
                    "macro_f1": stats["macro_f1"],
                    "kappa": stats["cohens_kappa"],
                })

            all_results.append(result_entry)

    if len(evaluated_summaries) > 1:
        print("=== Summary Across Evaluated Recordings ===")
        print(f"{'Patient':<16} {'Variant':<7} {'Ch':>3} {'Right/Total':>13} {'Acc':>8} {'MacroF1':>8} {'Kappa':>8}")
        for s in evaluated_summaries:
            rt = f"{s['correct']}/{s['total']}"
            print(
                f"{s['patient_id']:<16} {s['variant']:<7} {s['channels']:>3} "
                f"{rt:>13} {s['accuracy']*100:>7.2f}% {s['macro_f1']:>8.4f} {s['kappa']:>8.4f}"
            )

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! Inference results saved to: {args.output}")
