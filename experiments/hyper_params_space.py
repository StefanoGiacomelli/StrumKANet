import os
import itertools
import yaml
import pandas as pd


OUTPUT_DIR = "./experiments/configs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# HP Search Space ----------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
epochs = 1

task = ["bpm", "pattern"]
label_type = ["bpm", "pattern"]

bpm_source = "bpm_marco"

features_1d = [None, "onset_strength", "beats", "ticks", "bpm_estimates", "rh"]
features_2d = ["tempogram", "tempogram_ratio", "rp", "tssd", "trh"]

preprocessor = [None, True]

segment_duration = 10

loss = "mae"
cross_entropy_params = {"tolerance": [0.3],
                        "use_smoothing": [True, False]}

encoder_1d = {"out_chs": [16, 32],
              "kernel_size": [3, 5, 7],
              "out_size": 4096,
              "attention_type": [None, "posenc", "se", "convattn"]}

encoder_2d = {"out_chs": [16, 32],
              "kernel_size": [(3, 3), (5, 5), (7, 7)],
              "out_size": (64, 64),
              "attention_type": [None, "axial", "mhsa"]}

attention = {"positional_encoding_1d": {},
             "squeeze_excite_1d": {"reduction": 16},
             "conv_attention_1d": {"kernel_size": 3},
             "axial_attention_2d": {"heads": 8, "dim_head": 32},
             "mhsa_2d": {"heads": 8, "dim_head": 32}}

kan = {"type": ["BPMKAN", "PatternKAN"]}


# --------------------------------------------------------------------------------------------------------------------
def expand_encoder(encoder_dict):
    keys = [k for k in encoder_dict if isinstance(encoder_dict[k], list)]
    static_keys = {k: v for k, v in encoder_dict.items() if k not in keys}
    values = [encoder_dict[k] for k in keys]
    for combination in itertools.product(*values):
        combo_dict = dict(zip(keys, combination))
        attention_type = combo_dict.pop("attention_type", None)
        yield {**static_keys, **combo_dict}, attention_type

def convert_tuples_to_lists(d):
    if isinstance(d, dict):
        return {k: convert_tuples_to_lists(v) for k, v in d.items()}
    elif isinstance(d, (tuple, list)):
        return [convert_tuples_to_lists(v) for v in d]
    else:
        return d

def get_attention_params(attention_type, feature_type):
    if not attention_type:
        return None
    if feature_type == "1d":
        if attention_type == "posenc":
            return attention.get("positional_encoding_1d", {})
        elif attention_type == "se":
            return attention.get("squeeze_excite_1d", {})
        elif attention_type == "convattn":
            return attention.get("conv_attention_1d", {})
    elif feature_type == "2d":
        if attention_type == "axial":
            return attention.get("axial_attention_2d", {})
        elif attention_type == "mhsa":
            return attention.get("mhsa_2d", {})
    return {}

def get_loss_section(task_name):
    if task_name == "bpm":
        return {"name": loss}
    elif task_name == "pattern":
        loss_list = []
        for tol, smoothing in itertools.product(cross_entropy_params["tolerance"], cross_entropy_params["use_smoothing"]):
            loss_list.append({"name": "categorical_cross_entropy",
                              "kwargs": {"tolerance": tol,
                                         "use_smoothing": smoothing}})
        return loss_list
    else:
        raise ValueError(f"Unknown task: {task_name}")

def generate_configs():
    for task_name in task:
        for feat in features_1d + features_2d:
            # Determine the type of feature: 1D, 2D, or raw (None --> treated as 1D)
            if feat in features_2d:
                feature_type = "2d"
                encoder_defs = encoder_2d
                encoder_type = "2D"
            else:
                feature_type = "1d"
                encoder_defs = encoder_1d
                encoder_type = "1D"

            for encoder_params, attn_type in expand_encoder(encoder_defs):
                for kan_type in kan["type"]:
                    # Ensure that task and KAN type are matched correctly
                    if (task_name == "bpm" and kan_type != "BPMKAN") or (task_name == "pattern" and kan_type != "PatternKAN"):
                        continue

                    loss_section = get_loss_section(task_name)

                    # Configuration generation for "pattern" task (multi-loss)
                    if task_name == "pattern":
                        for loss_params in loss_section:
                            encoder_params = convert_tuples_to_lists(encoder_params)
                            config = {"task": task_name,
                                      "data": {"segment_duration": segment_duration,
                                               "label_type": task_name,
                                               "bpm_source": bpm_source,
                                               "features": [feat] if feat is not None else None,
                                               "preprocessor": True if feat is not None else None},
                                      "loss": loss_params,
                                      "encoder": {"type": encoder_type,
                                                  "params": encoder_params,
                                                  "attention": {"type": attn_type,
                                                                "params": get_attention_params(attn_type, feature_type)}},
                                      "kan": {"type": kan_type},
                                      "epochs": epochs}

                            # Compose filename
                            fname_parts = [f"ENC_chs{encoder_params['out_chs']}_ks_{encoder_params['kernel_size']}",
                                           f"att_{attn_type}",
                                           f"smooth_{loss_params.get('kwargs', {}).get('use_smoothing', '')}",
                                           feat if feat else "raw"]
                            filename = "_".join(str(p) for p in fname_parts if p is not None) + ".yaml"
                            out_dir = os.path.join(OUTPUT_DIR, task_name)
                            os.makedirs(out_dir, exist_ok=True)

                            # Add metadata fields
                            experiment_id = os.path.splitext(os.path.basename(filename))[0]
                            config["experiment_id"] = experiment_id
                            config["description"] = f"Automatically generated, see hyper_params_space.py for details"

                            # Write YAML file
                            with open(os.path.join(out_dir, filename), "w") as f:
                                yaml.dump(config, f, sort_keys=False)

                    # Configuration generation for "bpm" task (single loss)
                    else:
                        encoder_params = convert_tuples_to_lists(encoder_params)
                        config = {"task": task_name,
                                  "data": {"segment_duration": segment_duration,
                                           "label_type": task_name,
                                           "bpm_source": bpm_source,
                                           "features": [feat] if feat is not None else None,
                                           "preprocessor": True if feat is not None else None},
                                  "loss": loss_section,
                                  "encoder": {"type": encoder_type,
                                              "params": encoder_params,
                                              "attention": {"type": attn_type,
                                                            "params": get_attention_params(attn_type, feature_type)}},
                                  "kan": {"type": kan_type},
                                  "epochs": epochs}

                        # Compose filename
                        loss_name = loss_section["name"]
                        fname_parts = [f"ENC_chs{encoder_params['out_chs']}_ks_{encoder_params['kernel_size']}",
                                       f"att_{attn_type}",
                                       loss_name,
                                       feat if feat else "raw"]
                        filename = "_".join(str(p) for p in fname_parts if p is not None) + ".yaml"
                        out_dir = os.path.join(OUTPUT_DIR, task_name)
                        os.makedirs(out_dir, exist_ok=True)

                        # Add metadata fields
                        experiment_id = os.path.splitext(os.path.basename(filename))[0]
                        config["experiment_id"] = experiment_id
                        config["description"] = f"Automatically generated, see hyper_params_space.py for details"

                        # Write YAML file
                        with open(os.path.join(out_dir, filename), "w") as f:
                            yaml.dump(config, f, sort_keys=False)


# Run ----------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    from math import prod

    def count_expected_configs():
        total_count = {"bpm": 0, "pattern": 0}

        # Define full feature list including raw input (None), to be treated as 1D
        all_feats = features_1d + features_2d  # features_1d already includes None

        for task_name in task:
            for feat in all_feats:
                # Determine encoder type: 2D only if feature is in features_2d
                if feat in features_2d:
                    encoder_defs = encoder_2d
                else:  # includes raw input (None) and all 1D features
                    encoder_defs = encoder_1d

                # Compute all combinations of encoder hyperparameters
                encoder_keys = [k for k in encoder_defs if isinstance(encoder_defs[k], list)]
                encoder_combos = prod(len(encoder_defs[k]) for k in encoder_keys)

                if task_name == "bpm":
                    total_count["bpm"] += encoder_combos
                elif task_name == "pattern":
                    loss_combos = len(cross_entropy_params["tolerance"]) * len(cross_entropy_params["use_smoothing"])
                    total_count["pattern"] += encoder_combos * loss_combos

        print(f"[INFO] Expected configuration count:")
        for k, v in total_count.items():
            print(f" - {k}: {v} combinations")

    def count_generated_files():
        print(f"[INFO] YAML files actually generated:")
        for k in task:
            dir_path = os.path.join(OUTPUT_DIR, k)
            num_files = len(glob.glob(os.path.join(dir_path, "*.yaml")))
            print(f" - {dir_path}: {num_files} YAML files")

    count_expected_configs()
    generate_configs()
    count_generated_files()


    # Results sanity check ---------------------------------------------------------------------------------------------
    CONFIGS_DIR = "./experiments/configs/pattern/"  # Change to "./experiments/configs/bpm/" if needed
    RESULTS_DIR = "./experiments/results/"
    csv_path = os.path.join(RESULTS_DIR, "hp_search_results.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # STEP 1: Get all experiment IDs that have a corresponding YAML file
        yaml_files = glob.glob(os.path.join(CONFIGS_DIR, "*.yaml"))
        existing_ids = {os.path.splitext(os.path.basename(f))[0] for f in yaml_files}
        print(f"[DEBUG] YAML config files found: {len(existing_ids)}")

        # STEP 2: Keep only rows with experiment_id matching an existing YAML file
        df = df[df["experiment_id"].isin(existing_ids)]
        print(f"[DEBUG] Rows after YAML filtering: {len(df)}")

        # STEP 3: For each experiment_id, keep only the rows where test_status == True
        # If there are no such rows, the experiment_id is discarded entirely
        def filter_group(group):
            tested = group[group["test_status"] == True]
            return tested if not tested.empty else pd.DataFrame()

        df_filtered = df.groupby("experiment_id", group_keys=False).apply(filter_group)

        # STEP 4: Save the cleaned CSV
        df_filtered.to_csv(csv_path, index=False)
        print(f"[INFO] Cleaned {csv_path}: {len(df_filtered)} rows remaining.")
    else:
        print(f"[INFO] No {csv_path} file found to clean.")