"""Dataset-free smoke tests of StrumKANet (synthetic tensors, synthetic audio and stored files only).

Run from anywhere:  python tests/test_smoke.py        (or: python -m unittest discover -s tests)
They need no dataset, no checkpoints, no rp_extract and no Essentia. Synthetic recordings are written to a temporary folder.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402
import yaml                                          # noqa: E402
from pytorch_lightning import Trainer                # noqa: E402

import data_processing                               # noqa: E402
from analysis.common import CONFIG_DIR, STORED_RESULTS, read_csv_dicts, sanitize_id   # noqa: E402
from analysis.model_complexity import INPUT_SHAPES, lightning_count, parse_summary_table   # noqa: E402
from data import PopMusicDataModule, PopMusicDataset   # noqa: E402
from evaluation import (ProtocolOptions, class_report, cluster_bootstrap_ci, confusion_matrix, recording_of,   # noqa: E402
                        save_test_outputs)
from main_training import load_completed_experiments, sanitize_experiment_id   # noqa: E402
from model import Model, PatternKAN, init_modules    # noqa: E402
from splits import check_partition, grouped_kfold, grouped_partition   # noqa: E402
from WavKAN import KANLinear                         # noqa: E402

SEGMENT_SECONDS = 0.5                               # synthetic excerpts: 0.5 s windows of 3 s recordings (4 excerpts per recording)


def stored_config(experiment_id: str) -> dict:
    with open(CONFIG_DIR / "pattern" / f"{experiment_id}.yaml") as handle:
        return yaml.safe_load(handle)


def make_synthetic_dataset(root: Path, per_class: int = 10, seconds: int = 3, rate: int = 48000, seed: int = 0, performer_of=None, url_of=None):
    """Write ``3 * per_class`` synthetic stereo 48 kHz recordings (noise plus class-dependent clicks) and a dataset.csv; return (csv, audio dir).

    ``performer_of(label, k)`` and ``url_of(label, k)`` customise the CSV (default: one performer per recording, empty URLs)."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    audio_dir = root / "file_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    periods = {0: 0.25, 1: 0.125, 2: 1 / 3}
    for label in range(3):
        for k in range(per_class):
            performer, title = (performer_of(label, k) if performer_of else f"Band {label}{k:02d}"), f"Song {label}{k:02d}"
            signal = 0.05 * rng.standard_normal((seconds * rate, 2))
            for click in np.arange(0.0, seconds, periods[label]):
                signal[int(click * rate):int(click * rate) + 200] += 0.5
            sf.write(audio_dir / f"{performer.replace(' ', '_')}-{title.replace(' ', '_')}.wav", signal.astype("float32"), rate, subtype="PCM_16")
            rows.append({"title": title, "performer": performer, "yt_url": url_of(label, k) if url_of else "", "bpm_marco": 100, "label": label, "downloaded": "True"})
    csv_path = root / "dataset.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path), str(audio_dir) + "/"


class TestHeadAndWavKAN(unittest.TestCase):
    def test_head_parameter_counts(self):
        """12 D + 6 parameters (scale, translation, wavelet_weights, weight1, BatchNorm): 786,438 / 1,572,870."""
        for embedding, expected in ((65536, 786438), (131072, 1572870)):
            head = PatternKAN(in_feats=embedding, out_probs=3)
            self.assertEqual(sum(p.numel() for p in head.parameters()), expected)
            self.assertEqual(expected, 12 * embedding + 6)

    def test_weight1_is_registered_but_unused(self):
        """weight1 stays in the module (released checkpoints contain it) but receives no gradient."""
        layer = KANLinear(in_features=8, out_features=3)
        self.assertIn("weight1", dict(layer.named_parameters()))
        layer(torch.randn(4, 8)).sum().backward()
        self.assertIsNone(layer.weight1.grad)
        self.assertIsNotNone(layer.wavelet_weights.grad)

    def test_wavkan_unit_matches_closed_form(self):
        """y_j = BN( sum_i w_ji * psi((x_i - t_ji) / s_ji) ),  psi(u) = 2/(sqrt(3) pi^(1/4)) (u^2 - 1) exp(-u^2/2)."""
        torch.manual_seed(0)
        layer = KANLinear(in_features=6, out_features=3)
        with torch.no_grad():
            layer.scale.copy_(0.5 + torch.rand_like(layer.scale))
            layer.translation.copy_(torch.randn_like(layer.translation))
        x = torch.randn(5, 6)
        u = (x[:, None, :] - layer.translation[None]) / layer.scale[None]
        psi = 2 / (math.sqrt(3) * math.pi ** 0.25) * (u ** 2 - 1) * torch.exp(-0.5 * u ** 2)
        expected = (psi * layer.wavelet_weights[None]).sum(dim=2)
        self.assertTrue(torch.allclose(layer.wavelet_transform(x), expected, atol=1e-6))

    def test_kan_config_keyword_is_accepted(self):
        """init_modules already passed kan_config to constructors that did not accept it (latent TypeError)."""
        self.assertEqual(sum(p.numel() for p in PatternKAN(8, 3, kan_config={"wavelet_type": "mexican_hat"}).parameters()), 12 * 8 + 6)
        PatternKAN(8, 3, kan_config={"wavelet_type": "morlet"})
        with self.assertRaises(ValueError):
            PatternKAN(8, 3, kan_config={"unknown": 1})

    def test_ablation_heads_are_selectable_from_a_configuration(self):
        config = stored_config("ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp")
        x = torch.randn(4, 1, 24, 60)
        counts = {}
        for name, params in (("mlp", {"hidden_dims": [64, 32]}), ("linear", {"hidden_dims": []})):
            cfg = dict(config, kan={"type": "PatternFC", "params": params})
            encoder, head = init_modules(cfg, x)
            self.assertEqual(tuple(head(x, encoder=encoder).shape), (4, 3))
            counts[name] = sum(p.numel() for p in head.parameters())
        self.assertEqual(counts["linear"], 65536 * 3 + 3)


class TestLoss(unittest.TestCase):
    @staticmethod
    def _model(smoothing: bool) -> Model:
        config = {"loss": {"name": "categorical_cross_entropy", "kwargs": {"tolerance": 0.3, "use_smoothing": smoothing}}}
        return Model(torch.nn.Identity(), torch.nn.Identity(), downstream_task="pattern", config=config)

    def test_smoothed_loss_is_the_tolerance_likelihood(self):
        """L = -log sum_j M[y, j] p_j (hard label); NOT the soft-target cross-entropy."""
        torch.manual_seed(1)
        logits, labels = torch.randn(64, 3), torch.randint(0, 3, (64,))
        p = F.softmax(logits, dim=1)
        tau = 0.3
        likelihood = torch.where(labels == 2, p[:, 2], (1 - tau) * p.gather(1, labels[:, None])[:, 0]
                                 + tau * p.gather(1, (1 - labels.clamp(max=1))[:, None])[:, 0])
        expected = -torch.log(likelihood + 1e-8).mean()
        self.assertAlmostEqual(self._model(True).loss_fn(logits, labels).item(), expected.item(), places=5)

    def test_unsmoothed_loss_is_cross_entropy(self):
        logits, labels = torch.randn(32, 3), torch.randint(0, 3, (32,))
        self.assertAlmostEqual(self._model(False).loss_fn(logits, labels).item(), F.cross_entropy(logits, labels).item(), places=6)


class TestStoredModels(unittest.TestCase):
    def test_14_stored_models_match_the_logged_summary(self):
        """Instantiate the 14 Stage-2 configurations: parameter counts equal the Lightning summary of training_log.txt."""
        summary = parse_summary_table((STORED_RESULTS / "training_log.txt").read_text(errors="replace"))
        stored = read_csv_dicts(STORED_RESULTS / "training_results.csv")
        self.assertEqual(len(stored), 14)
        for record in stored:
            experiment_id = record["experiment_id"]
            config = stored_config(experiment_id)
            descriptor = config["data"]["features"][0]
            torch.manual_seed(0)
            encoder, head = init_modules(config, torch.randn(2, 1, *INPUT_SHAPES[descriptor]))
            cells = summary[sanitize_id(experiment_id)]
            self.assertEqual(lightning_count(sum(p.numel() for p in encoder.parameters())), cells["encoder"], experiment_id)
            self.assertEqual(lightning_count(sum(p.numel() for p in head.parameters())), cells["kan"], experiment_id)

    def test_forward_shapes_of_one_encoder_per_family(self):
        for experiment_id, shape in (("ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp", (24, 60)),
                                     ("ENC_chs16_ks_7_att_convattn_smooth_False_onset_strength", (938,)),
                                     ("ENC_chs32_ks_[7, 7]_att_mhsa_smooth_False_tempogram_ratio", (13, 938)),
                                     ("ENC_chs16_ks_5_att_convattn_smooth_True_rh", (60,))):
            config = stored_config(experiment_id)
            x = torch.randn(3, 1, *shape)
            encoder, head = init_modules(config, x)
            self.assertEqual(tuple(head(x, encoder=encoder).shape), (3, 3), experiment_id)

    def test_grid_files_are_all_present(self):
        self.assertEqual(len(list((CONFIG_DIR / "pattern").glob("*.yaml"))), 468)
        self.assertEqual(len(list((CONFIG_DIR / "bpm").glob("*.yaml"))), 234)      # generated, never run (kept, documented)
        self.assertEqual(len(read_csv_dicts(STORED_RESULTS / "hp_search_results.csv")), 468)

    def test_completed_runs_are_skipped_with_sanitized_ids(self):
        """Fix: the results CSV keeps '[7, 7]' while the loop compares '7-7'; both sides are now sanitized."""
        stored = load_completed_experiments(str(STORED_RESULTS / "training_results.csv"))
        sanitized = {sanitize_experiment_id(i) for i in stored}
        self.assertEqual(len(sanitized), 14)
        two_d = sanitize_experiment_id("ENC_chs16_ks_[7, 7]_att_None_smooth_True_rp")
        self.assertNotIn(two_d, stored)             # the legacy comparison could never match 2D runs
        self.assertIn(two_d, sanitized)


class TestRegeneration(unittest.TestCase):
    def test_regeneration_chain_reproduces_the_published_checks(self):
        """analysis/ rebuilds the tables from the stored files; every published-value check must pass."""
        from analysis import stage1_grid, stage2_table
        with tempfile.TemporaryDirectory() as tmp:
            for step in (stage1_grid, stage2_table):
                record = step.run(Path(tmp))
                failed = [c for c in record["checks"] if not c["pass"]]
                self.assertEqual(failed, [], record["script"])
            self.assertTrue((Path(tmp) / "stage2_table.csv").is_file())


class TestPartitionUtilities(unittest.TestCase):
    @staticmethod
    def _corpus(per_class=10, excerpts=4):
        groups, labels = [], []
        for label in range(3):
            for k in range(per_class):
                groups += [f"rec{label}{k:02d}"] * excerpts
                labels += [label] * excerpts
        return groups, labels

    def test_grouped_partition_is_recording_disjoint_stratified_and_deterministic(self):
        groups, labels = self._corpus()
        parts = grouped_partition(groups, labels, (0.8, 0.1, 0.1), seed=1)
        summary = check_partition(parts, groups, len(groups))             # raises if a recording is shared or an excerpt is missing
        self.assertEqual(summary["recordings_in_more_than_one_part"], 0)
        self.assertEqual(summary["excerpts"], [96, 12, 12])
        for part in parts:
            self.assertEqual({labels[i] for i in part}, {0, 1, 2})        # every part contains every class
        self.assertEqual(parts, grouped_partition(groups, labels, (0.8, 0.1, 0.1), seed=1))
        self.assertNotEqual(parts, grouped_partition(groups, labels, (0.8, 0.1, 0.1), seed=2))

    def test_grouped_kfold_holds_every_recording_out_exactly_once(self):
        groups, labels = self._corpus()
        folds = grouped_kfold(groups, labels, 5, seed=3)
        check_partition(folds, groups, len(groups))
        self.assertEqual([len({groups[i] for i in f}) for f in folds], [6] * 5)
        self.assertEqual(sorted({g for f in folds for g in {groups[i] for i in f}}), sorted(set(groups)))

    def test_label_must_be_constant_within_a_recording(self):
        with self.assertRaises(ValueError):
            grouped_partition(["a", "a", "b", "b"], [0, 1, 1, 1], (0.5, 0.5), seed=0)


class TestEvaluationHelpers(unittest.TestCase):
    def test_confusion_matrix_and_report(self):
        cm = confusion_matrix([0, 1, 2, 2], [0, 2, 2, 2])
        self.assertEqual(cm.tolist(), [[1, 0, 0], [0, 0, 1], [0, 0, 2]])
        report = class_report(cm)
        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["balanced_accuracy"], (1 + 0 + 1) / 3)

    def test_cluster_bootstrap_resamples_recordings(self):
        groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
        values = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
        low, high = cluster_bootstrap_ci(groups, values, resamples=500, seed=0)
        self.assertLess(low, statistics.mean(values))
        self.assertGreater(high, statistics.mean(values))
        self.assertEqual(recording_of("Band_000-Song_000_seg3"), "Band_000-Song_000")

    def test_default_protocol_options_are_the_stored_protocol(self):
        legacy = ProtocolOptions()
        self.assertEqual((legacy.split_mode, legacy.eval_ckpt, legacy.accelerator, legacy.save_predictions), ("legacy", "last", "auto", False))
        self.assertEqual(legacy.namespace(), "")                         # stored files are written where they always were
        self.assertEqual(ProtocolOptions(split_mode="grouped", eval_ckpt="best").namespace(), "grouped_seed42_best")


class TestDataProcessing(unittest.TestCase):
    def test_rp_extract_is_only_required_when_requested(self):
        if data_processing.RP_EXTRACT_AVAILABLE:
            self.skipTest("rp_extract is installed")
        with self.assertRaises(ImportError) as context:
            data_processing.rhythm_features(np.zeros(44100, dtype=np.float32), extractors=["rp_extract"])
        self.assertIn("rp_extract", str(context.exception))
        features = data_processing.rhythm_features(np.random.default_rng(0).standard_normal(44100 * 2).astype(np.float32), extractors=["librosa"])
        self.assertIn("onset_strength", features["librosa"])


class TestSyntheticDatasetProtocols(unittest.TestCase):
    """The repository's own data module on synthetic audio: the legacy split is re-drawn; the new options draw once."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.csv_path, cls.audio = make_synthetic_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _module(self, **options) -> PopMusicDataModule:
        return PopMusicDataModule(csv_path=self.csv_path, audio_folder=self.audio, segment_duration=SEGMENT_SECONDS, label_type="pattern",
                                  split_ratio=(0.8, 0.1, 0.1), batch_size=8, num_workers=0, **options)

    def test_dataset_geometry(self):
        dataset = PopMusicDataset(self.csv_path, self.audio, segment_duration=SEGMENT_SECONDS, label_type="pattern")
        self.assertEqual(len(dataset), 30 * 4)                           # windows 1..4 of 6 (first and last dropped)
        self.assertEqual(tuple(dataset[0]["input"].shape), (2, int(SEGMENT_SECONDS * 48000)))

    def test_legacy_split_is_redrawn_on_every_setup_call(self):
        """Mechanism of the validation/test gap: each setup() call returns a new permutation, so about 80 % of a re-drawn test set was training data."""
        torch.manual_seed(0)
        module = self._module()
        overlaps, previous_train, previous_test = [], None, None
        for _ in range(60):
            module.setup()
            train, test = set(module.train_set.indices), set(module.test_set.indices)
            if previous_train is not None:
                overlaps.append(len(previous_train & test) / len(test))
                self.assertNotEqual(previous_test, test)
            previous_train, previous_test = train, test
        self.assertGreater(statistics.mean(overlaps), 0.72)
        self.assertLess(statistics.mean(overlaps), 0.88)                 # expectation 0.80 (hypergeometric)
        self.assertEqual((len(module.train_set), len(module.val_set), len(module.test_set)), (96, 12, 12))

    def test_seeded_partition_is_drawn_once(self):
        module = self._module(split_mode="seeded", split_seed=7)
        module.setup()
        first = [list(s.indices) for s in (module.train_set, module.val_set, module.test_set)]
        for stage in ("fit", "test", None):
            module.setup(stage)
            self.assertEqual([list(s.indices) for s in (module.train_set, module.val_set, module.test_set)], first)
        other = self._module(split_mode="seeded", split_seed=7)
        other.setup()
        self.assertEqual(first[2], list(other.test_set.indices))
        different = self._module(split_mode="seeded", split_seed=8)
        different.setup()
        self.assertNotEqual(first[2], list(different.test_set.indices))

    def test_grouped_partition_is_recording_disjoint_and_stable(self):
        module = self._module(split_mode="grouped", split_seed=5)
        module.setup()
        groups, labels = module.dataset.partition_keys()
        recordings = [{groups[i] for i in s.indices} for s in (module.train_set, module.val_set, module.test_set)]
        self.assertFalse(recordings[0] & recordings[1] or recordings[0] & recordings[2] or recordings[1] & recordings[2])
        self.assertEqual(module.partition_summary["recordings_in_more_than_one_part"], 0)
        self.assertEqual([len(r) for r in recordings], [24, 3, 3])
        test_before = list(module.test_set.indices)
        module.setup("test")
        self.assertEqual(test_before, list(module.test_set.indices))

    def test_partition_file_is_stored_and_verified(self):
        path = str(self.root / "partition.json")
        first = self._module(split_mode="grouped", split_seed=5, partition_file=path)
        first.setup()
        self.assertTrue(Path(path).is_file())
        self._module(split_mode="grouped", split_seed=5, partition_file=path).setup()          # same partition: accepted
        with self.assertRaises(ValueError):
            self._module(split_mode="grouped", split_seed=6, partition_file=path).setup()      # other seed: refused

    def test_dataloaders_work_without_worker_processes(self):
        module = self._module(split_mode="seeded", split_seed=1)
        module.setup()
        self.assertEqual(len(next(iter(module.val_dataloader()))["id"]), 8)                  # persistent_workers=True used to raise here

    def test_predictions_and_confusion_matrix_are_saved(self):
        module = self._module(split_mode="grouped", split_seed=3)
        module.setup()
        config = {"loss": {"name": "categorical_cross_entropy", "kwargs": {"tolerance": 0.3, "use_smoothing": False}},
                  "encoder": {"type": "1D", "params": {"out_chs": 4, "kernel_size": 3, "out_size": 16}, "attention": {"type": None, "params": None}},
                  "kan": {"type": "PatternKAN"}}
        encoder, head = init_modules(config, next(iter(module.train_dataloader()))["input"])
        model = Model(encoder, head, downstream_task="pattern", config=config)
        trainer = Trainer(accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, limit_train_batches=3)
        trainer.fit(model, datamodule=module)
        trainer.test(model, datamodule=module, verbose=False)
        outputs = model.test_outputs_npz
        self.assertEqual(len(outputs["ids"]), len(module.test_set))
        with tempfile.TemporaryDirectory() as out:
            metrics = save_test_outputs(outputs, out, "run")
            self.assertEqual(sum(sum(r) for r in metrics["confusion_matrix"]), len(module.test_set))
            self.assertEqual(metrics["n_recordings"], 3)
            for suffix in ("_predictions.csv", "_confusion_matrix.csv", "_metrics.json"):
                self.assertTrue((Path(out) / f"run{suffix}").is_file())


class TestRecordingDisjointPipeline(unittest.TestCase):
    """End-to-end run of main_recording_disjoint.py on synthetic audio (2 epochs, 3 folds): plumbing, not scientific content."""

    def test_pipeline_on_synthetic_audio(self):
        import main_recording_disjoint as t9
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, audio = make_synthetic_dataset(root / "data")
            config = {"task": "pattern", "data": {"features": ["onset_strength"], "preprocessor": True, "label_type": "pattern"},
                      "loss": {"name": "categorical_cross_entropy", "kwargs": {"tolerance": 0.3, "use_smoothing": False}},
                      "encoder": {"type": "1D", "params": {"out_chs": 4, "kernel_size": 3, "out_size": 16}, "attention": {"type": None, "params": None}},
                      "kan": {"type": "PatternKAN"}}
            result = t9.run_recording_disjoint({"synthetic_os": config}, csv_path, audio, str(root / "cache"), str(root / "out"), str(root / "ckpt"),
                                               folds=3, seed=1, heads=("kan", "linear"), baselines=True, segment_duration=SEGMENT_SECONDS,
                                               accelerator="cpu", max_epochs=2, patience=1, batch_size=8, bootstrap=100)
            methods = {s["method"]: s for s in result["summaries"]}
            self.assertEqual(set(methods), {"synthetic_os|head=kan", "synthetic_os|head=linear",
                                            "baseline:majority|descriptor=onset_strength", "baseline:logreg|descriptor=onset_strength"})
            for summary in methods.values():
                self.assertEqual((summary["n_excerpts"], summary["n_recordings"]), (120, 30))      # every recording held out exactly once
                self.assertLessEqual(summary["accuracy_ci_cluster_low"], summary["accuracy"])
                self.assertGreaterEqual(summary["accuracy_ci_cluster_high"], summary["accuracy"])
            with open(result["partition"]) as handle:
                partition = json.load(handle)
            for part in partition["parts"]:
                self.assertFalse(set(part["train"]) & set(part["test"]) or set(part["val"]) & set(part["test"]) or set(part["train"]) & set(part["val"]))
            for name in ("summary.csv", "folds.csv", "confusion_matrices.json", "PROVENANCE.json"):
                self.assertTrue((root / "out" / name).is_file(), name)

    @staticmethod
    def _tiny_configuration() -> dict:
        return {"task": "pattern", "data": {"features": ["onset_strength"], "preprocessor": True, "label_type": "pattern"},
                "loss": {"name": "categorical_cross_entropy", "kwargs": {"tolerance": 0.3, "use_smoothing": False}},
                "encoder": {"type": "1D", "params": {"out_chs": 4, "kernel_size": 3, "out_size": 16}, "attention": {"type": None, "params": None}},
                "kan": {"type": "PatternKAN"}}

    def test_parallel_workers_train_every_run_once_and_resume(self):
        """--workers: child processes claim runs, every run file exists exactly once, a second call only aggregates (nothing is retrained)."""
        import main_recording_disjoint as t9
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, audio = make_synthetic_dataset(root / "data")
            arguments = dict(folds=3, seed=1, heads=("kan", "linear"), baselines=False, segment_duration=SEGMENT_SECONDS, accelerator="cpu", max_epochs=2,
                             patience=1, batch_size=8, bootstrap=50, workers=2, threads=1, no_nnpack=True)
            result = t9.run_recording_disjoint({"synthetic_os": self._tiny_configuration()}, csv_path, audio, str(root / "cache"), str(root / "out"), str(root / "ckpt"),
                                               **arguments)
            runs = sorted((root / "out" / "runs").glob("*.json"))
            self.assertEqual(len(runs), 3 * 2)                                           # 3 folds x 2 heads, each trained once
            workers_used = {json.loads(path.read_text())["worker"] for path in runs}
            self.assertTrue(workers_used <= {"worker0", "worker1"} and workers_used)
            self.assertEqual({s["method"]: s["n_excerpts"] for s in result["summaries"]}, {"synthetic_os|head=kan": 120, "synthetic_os|head=linear": 120})
            before = {path.name: path.stat().st_mtime_ns for path in runs}
            t9.run_recording_disjoint({"synthetic_os": self._tiny_configuration()}, csv_path, audio, str(root / "cache"), str(root / "out"), str(root / "ckpt"), **arguments)
            self.assertEqual(before, {path.name: path.stat().st_mtime_ns for path in sorted((root / "out" / "runs").glob("*.json"))})   # nothing retrained
            with open(root / "out" / "PROVENANCE.json") as handle:
                provenance = json.load(handle)
            self.assertEqual(provenance["parameters"]["workers"], 2)
            self.assertIn("main_recording_disjoint.py", provenance["code_sha256"])
            self.assertEqual(len(provenance["runs"]), 6)

    def test_excerpt_level_control_shares_recordings_between_training_and_test(self):
        """--grouping excerpt: every excerpt is held out once, but the recordings of held-out excerpts also appear in training (the control)."""
        import main_recording_disjoint as t9
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, audio = make_synthetic_dataset(root / "data")
            result = t9.run_recording_disjoint({"synthetic_os": self._tiny_configuration()}, csv_path, audio, str(root / "cache"), str(root / "out"), str(root / "ckpt"),
                                               folds=3, seed=1, heads=("linear",), baselines=False, segment_duration=SEGMENT_SECONDS, accelerator="cpu",
                                               max_epochs=1, patience=1, batch_size=8, bootstrap=50, grouping="excerpt")
            self.assertEqual(result["summaries"][0]["n_excerpts"], 120)
            with open(result["partition"]) as handle:
                partition = json.load(handle)
            self.assertEqual(partition["grouping"], "excerpt")
            for part in partition["parts"]:
                self.assertFalse(set(part["train"]) & set(part["test"]))                       # excerpts are disjoint ...
                self.assertTrue({k.rsplit("|", 1)[0] for k in part["train"]} & {k.rsplit("|", 1)[0] for k in part["test"]})   # ... recordings are not (key = recording|segment)
            with open(root / "out" / "PROVENANCE.json") as handle:
                self.assertTrue(json.load(handle)["NEW_RUN"].startswith("CONTROL"))


class TestRecordingDisjointReport(unittest.TestCase):
    """analysis/recording_disjoint_report.py on outputs of the synthetic pipeline: tables exist and are internally consistent."""

    def test_report_tables_on_synthetic_runs(self):
        import gzip
        import main_recording_disjoint as t9
        from analysis import recording_disjoint_report as report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, audio = make_synthetic_dataset(root / "data")
            config = TestRecordingDisjointPipeline._tiny_configuration()
            name = "ENC_synthetic_onset_strength"
            common = dict(folds=3, seed=1, segment_duration=SEGMENT_SECONDS, accelerator="cpu", max_epochs=2, patience=1, batch_size=8, bootstrap=100)
            runs = {}
            for grouping, heads in (("recording", ("kan", "linear")), ("excerpt", ("kan",))):
                out = root / grouping
                t9.run_recording_disjoint({name: config}, csv_path, audio, str(root / "cache"), str(out), str(root / "ckpt"), heads=heads, baselines=True, grouping=grouping, **common)
                runs[grouping] = out
            # predictions may be shipped gzip-compressed: the report must read them
            for path in (runs["recording"] / "predictions").glob("*.csv"):
                path.with_name(path.name + ".gz").write_bytes(gzip.compress(path.read_bytes()))
                path.unlink()
            out_dir = root / "report"
            result = report.run(runs["recording"], runs["excerpt"], None, [], out_dir, ROOT)
            for table in ("methods.csv", "families.csv", "ladder.csv", "paired.csv", "confusion_long.csv", "report.json", "methods_control_excerpt.csv"):
                self.assertTrue((out_dir / table).is_file(), table)
            methods = {r["method"]: r for r in report.read_csv(out_dir / "methods.csv")}
            self.assertEqual(set(methods), {f"{name}|head=kan", f"{name}|head=linear", "baseline:majority|descriptor=onset_strength", "baseline:logreg|descriptor=onset_strength"})
            paired = {(r["contrast"], r["method_a"]): r for r in report.read_csv(out_dir / "paired.csv")}
            difference = paired[("WavKAN head - majority class", f"{name}|head=kan")]
            self.assertAlmostEqual(float(difference["difference"]), float(methods[f"{name}|head=kan"]["accuracy"]) - float(methods["baseline:majority|descriptor=onset_strength"]["accuracy"]), places=5)
            self.assertLessEqual(float(difference["ci_low"]), float(difference["difference"]))
            self.assertGreaterEqual(float(difference["ci_high"]), float(difference["difference"]))
            self.assertTrue(any(c.startswith("recording-disjoint - excerpt-level folds") for c, _ in paired))
            confusion = [r for r in report.read_csv(out_dir / "confusion_long.csv") if r["protocol"] == "recording" and r["method"] == f"{name}|head=kan"]
            self.assertEqual(sum(int(r["count"]) for r in confusion), 120)                       # every excerpt is held out once
            self.assertEqual(result["report"]["recording"]["baselines"]["baseline:majority|descriptor=onset_strength"]["accuracy"], float(methods["baseline:majority|descriptor=onset_strength"]["accuracy"]))


class TestGroupingUnits(unittest.TestCase):
    """Grouping units of the recording-disjoint pipeline: repeated sources, performers with several classes, missing audio."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.csv_path, cls.audio = make_synthetic_dataset(cls.root, per_class=4, performer_of=lambda label, k: f"Band {k % 2}",
                                                          url_of=lambda label, k: "https://example.org/watch?v=same" if (label, k) in ((0, 0), (0, 1)) else "")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_recording_key_merges_rows_with_the_same_source(self):
        from feature_cache import grouping_keys, index_dataset
        dataset = index_dataset(self.csv_path, self.audio, SEGMENT_SECONDS)
        recordings, labels = dataset.partition_keys()
        keys = grouping_keys(dataset, "recording")
        self.assertEqual(len(set(recordings)), 12)
        self.assertEqual(len(set(keys)), 11)                                             # the two rows with the same URL are one recording
        by_file = {r: k for r, k in zip(recordings, keys)}
        self.assertEqual(by_file["Band_1-Song_001"], by_file["Band_0-Song_000"])               # same URL -> same group
        self.assertNotEqual(by_file["Band_0-Song_002"], by_file["Band_0-Song_000"])
        self.assertEqual(len(set(grouping_keys(dataset, "excerpt"))), len(set(keys)) * 4)  # 4 excerpts per recording (the merged source shares its excerpts' keys)

    def test_performer_folds_are_performer_disjoint_with_mixed_classes(self):
        from feature_cache import grouping_keys, index_dataset
        from splits import check_partition, mixed_group_kfold, mixed_group_partition
        dataset = index_dataset(self.csv_path, self.audio, SEGMENT_SECONDS)
        recordings, labels = dataset.partition_keys()
        performers = grouping_keys(dataset, "performer")
        self.assertEqual(sorted(set(performers)), ["band 0", "band 1"])
        self.assertTrue(all(len({labels[i] for i, p in enumerate(performers) if p == name}) == 3 for name in set(performers)))      # every performer has all 3 classes
        folds = mixed_group_kfold(performers, labels, 2, seed=3)
        summary = check_partition(folds, performers, len(performers))
        self.assertEqual(summary["recordings_in_more_than_one_part"], 0)
        self.assertEqual(mixed_group_kfold(performers, labels, 2, seed=3), folds)           # deterministic
        two = mixed_group_partition([f"g{i // 3}" for i in range(48)], [i % 3 for i in range(48)], (0.875, 0.125), seed=1)
        self.assertEqual(sorted(two[0] + two[1]), list(range(48)))
        self.assertEqual(len(two[1]), 6)                                                          # one eighth of the 16 groups = 2 groups = 6 excerpts

    def test_missing_audio_is_an_error_not_a_silent_loss(self):
        import shutil
        from feature_cache import missing_audio
        import main_recording_disjoint as t9
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, audio = make_synthetic_dataset(Path(tmp) / "data", per_class=2)
            self.assertEqual(missing_audio(csv_path, audio), [])
            (Path(audio) / "Band_100-Song_100.wav").unlink()
            self.assertEqual(missing_audio(csv_path, audio), ["Band_100-Song_100"])
            config = TestRecordingDisjointPipeline._tiny_configuration()
            with self.assertRaises(FileNotFoundError):
                t9.run_recording_disjoint({"synthetic_os": config}, csv_path, audio, str(Path(tmp) / "cache"), str(Path(tmp) / "out"), str(Path(tmp) / "ckpt"),
                                          folds=2, seed=1, segment_duration=SEGMENT_SECONDS, accelerator="cpu", max_epochs=1)


class TestVariants(unittest.TestCase):
    """main_variants.py (training-recipe and descriptor variants) and analysis/variants_report.py on synthetic data: plumbing, not scientific content."""

    @staticmethod
    def _rp_configuration() -> dict:
        return {"task": "pattern", "data": {"features": ["rp"], "preprocessor": True, "label_type": "pattern"},
                "loss": {"name": "categorical_cross_entropy", "kwargs": {"tolerance": 0.3, "use_smoothing": False}},
                "encoder": {"type": "2D", "params": {"out_chs": 4, "kernel_size": [3, 3], "out_size": [8, 8]}, "attention": {"type": None, "params": None}},
                "kan": {"type": "PatternKAN"}}

    @staticmethod
    def _synthetic_rp_cache(root: Path, csv_path: str, audio: str) -> str:
        """An RP-shaped cache (24 x 60, class-dependent modulation peak) for the synthetic recordings: no rp_extract needed."""
        from feature_cache import index_dataset
        dataset = index_dataset(csv_path, audio, SEGMENT_SECONDS, "pattern")
        ids = [dataset.excerpt_id(i) for i in range(len(dataset))]
        labels = dataset.partition_keys()[1]
        rng = np.random.default_rng(0)
        rp = 0.1 * np.abs(rng.standard_normal((len(ids), 1, 24, 60))).astype(np.float32)
        for i, label in enumerate(labels):
            rp[i, 0, :, 10 + 12 * label] += 1.0
        cache = root / "cache"
        cache.mkdir()
        np.save(cache / "rp.npy", rp)
        for name in ("rp_view1", "rp_view2"):
            np.save(cache / f"{name}.npy", (rp * rng.uniform(0.9, 1.1, rp.shape)).astype(np.float32))
        with open(cache / "index.json", "w") as handle:
            json.dump({"segment_duration": SEGMENT_SECONDS, "label_type": "pattern", "ids": ids}, handle)
        return str(cache)

    def test_variant_names_combine_and_unknown_names_fail(self):
        import main_variants as mv
        self.assertEqual(mv.parse_variant("base"), {})
        self.assertEqual(mv.parse_variant("bal+aug"), {"class_weight": "balanced", "augment": "geo"})
        with self.assertRaises(ValueError):
            mv.parse_variant("bal+nonsense")
        with self.assertRaises(ValueError):
            mv.parse_variant("aug+mask")                                      # both set "augment"

    def test_transforms_are_per_excerpt_and_the_log_axis_maps_frequencies(self):
        import main_variants as mv
        rp = np.abs(np.random.default_rng(1).standard_normal((6, 1, 24, 60))).astype(np.float32)
        for name in ("bandnorm", "l2", "logmod"):
            full = mv.transform_rp(rp, name)
            self.assertEqual(full.shape, rp.shape)
            self.assertTrue(np.isfinite(full).all())
            np.testing.assert_allclose(mv.transform_rp(rp[2:4], name), full[2:4], rtol=1e-6)      # no statistic is shared between excerpts
        ramp = np.tile((np.arange(60) + 1.0) * (10.0 / 60.0), (1, 1, 24, 1)).astype(np.float32)   # value = modulation frequency in Hz
        np.testing.assert_allclose(mv.transform_rp(ramp, "logmod")[0, 0, 0, :], 0.5 * 20.0 ** (np.arange(60) / 59.0), rtol=1e-4)

    def test_augmentations_keep_the_shape_and_move_a_peak_as_documented(self):
        import main_variants as mv
        x = torch.zeros(4, 1, 24, 60)
        x[:, :, :, 20] = 1.0
        scaled = mv.augment_geo(x, tempo=(1.08, 1.08), shift=0, gain=(1.0, 1.0))                  # tempo x 1.08: bin k <-> (k + 1) * 10 / 60 Hz moves to 1.08 f
        centroid = float((scaled[0, 0, 0] * torch.arange(60)).sum() / scaled[0, 0, 0].sum())
        self.assertAlmostEqual(centroid, 21 * 1.08 - 1, delta=0.1)
        for augmented in (mv.augment_geo(x), mv.augment_geo_log(x), mv.augment_mask(x)):
            self.assertEqual(augmented.shape, x.shape)
            self.assertTrue(torch.isfinite(augmented).all())

    def test_base_variant_reproduces_the_stored_pipeline_and_the_report_pairs_them(self):
        """The 'base' variant is main_recording_disjoint.py's recipe: same folds, same seeds, identical predictions; variants add runs next to it."""
        import gzip
        import os
        import main_recording_disjoint as t9
        import main_variants as mv
        from analysis import variants_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, audio = make_synthetic_dataset(root / "data")
            cache = self._synthetic_rp_cache(root, csv_path, audio)
            name = "ENC_chs16_ks_[5, 5]_att_None_smooth_False_rp"                                    # a stored configuration (main_variants.py loads them by id)
            from analysis.recording_disjoint_report import safe
            common = dict(folds=3, seed=1, segment_duration=SEGMENT_SECONDS, accelerator="cpu", max_epochs=2, patience=2, batch_size=8, bootstrap=50)
            t9.run_recording_disjoint({name: stored_config(name)}, csv_path, audio, cache, str(root / "stored"), str(root / "ckpt"), heads=("kan", "linear"), baselines=True,
                                      threads=1, **common)
            for path in (root / "stored" / "predictions").glob("*.csv"):                              # shipped files are gzip-compressed
                path.with_name(path.name + ".gz").write_bytes(gzip.compress(path.read_bytes()))
                path.unlink()
            cwd = os.getcwd()
            os.chdir(ROOT)                                                                           # configurations are read from ./experiments/configs/pattern
            try:
                mv.main(["--variants", "base", "bal+hier", "views", "logmod", "ens3", "--configs", name, "--heads", "linear", "--folds", "3", "--seed", "1",
                         "--csv", csv_path, "--audio", audio, "--cache-dir", cache, "--out", str(root / "variants"), "--checkpoints", str(root / "ckpt2"),
                         "--max-epochs", "2", "--patience", "2", "--batch-size", "8", "--threads", "1", "--bootstrap", "50", "--segment-duration", str(SEGMENT_SECONDS)])
                mv.main(["--classical", "lr_rp", "lr_rp_bal", "lr_rp_bandnorm", "--folds", "3", "--seed", "1", "--csv", csv_path, "--audio", audio, "--cache-dir", cache,
                         "--out", str(root / "variants"), "--bootstrap", "50", "--segment-duration", str(SEGMENT_SECONDS)])
            finally:
                os.chdir(cwd)
            rows = {r["method"]: r for r in variants_report.read_summary_rows(root / "variants" / "summary.csv")}
            self.assertEqual(len(rows), 5 + 3)
            for row in rows.values():
                self.assertEqual((int(row["n_excerpts"]), int(row["n_recordings"])), (120, 30))
            base_predictions = variants_report.read_csv(root / "variants" / "predictions" / f"{safe(name + '|head=linear|variant=base')}.csv.gz")
            reference = variants_report.read_csv(root / "stored" / "predictions" / f"{safe(name + '|head=linear')}.csv.gz")
            self.assertEqual([r["id"] for r in base_predictions], [r["id"] for r in reference])           # same folds, same order
            self.assertEqual([r["prediction"] for r in base_predictions], [r["prediction"] for r in reference])   # same recipe, same seeds
            out = root / "report"
            variants_report.main(["--variants", str(root / "variants"), "--stored", str(root / "stored"), "--out", str(out)])
            table = {r["method"]: r for r in variants_report.read_csv(out / "variants.csv")}
            self.assertAlmostEqual(float(table[f"{name}|head=linear|variant=base"]["vs_stored_same_head_difference"]), 0.0, places=9)
            self.assertIn("vs_stored_kan_difference", table[f"{name}|head=linear|variant=base"])
            self.assertIn("vs_stored_logreg_difference", table["classical:lr_rp|descriptor=rp"])
            self.assertTrue({"variants.csv", "variants_pooled.csv", "variants.md"} <= {p.name for p in out.iterdir()})
            with open(root / "variants" / "PROVENANCE.json") as handle:
                provenance = json.load(handle)
            self.assertIn("main_variants.py", provenance["code_sha256"])
            self.assertEqual(len(provenance["commands"]), 2)


class TestVectorModelsAndControls(unittest.TestCase):
    """main_variants.py: heads on standardized descriptor vectors, classical models, the label-permutation control (synthetic caches, plumbing only)."""

    def test_vector_heads_classical_models_and_permutation_control(self):
        import os
        import main_variants as mv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, audio = make_synthetic_dataset(root / "data")
            cache = TestVariants._synthetic_rp_cache(root, csv_path, audio)
            from feature_cache import index_dataset
            labels = np.array(index_dataset(csv_path, audio, SEGMENT_SECONDS, "pattern").partition_keys()[1])
            rng = np.random.default_rng(3)
            onset = np.abs(rng.standard_normal((len(labels), 1, 938))).astype(np.float32)
            ratio = np.abs(rng.standard_normal((len(labels), 1, 13, 938))).astype(np.float32) * 0.1
            ratio[:, 0, 3, :] += labels[:, None] * 0.5                                                # a class-dependent tempo-ratio band
            np.save(Path(cache) / "onset_strength.npy", onset)
            np.save(Path(cache) / "tempogram_ratio.npy", ratio)
            features = mv.classical_features(cache, ["lr_tr", "lr_osscale"])
            self.assertEqual(features["tr_stats"].shape, (len(labels), 26))
            self.assertEqual(features["os_scale"].shape, (len(labels), 32))
            self.assertTrue(np.isfinite(features["tr_stats"]).all() and np.isfinite(features["os_scale"]).all())
            cwd = os.getcwd()
            os.chdir(ROOT)
            try:
                common = ["--folds", "3", "--seed", "1", "--csv", csv_path, "--audio", audio, "--cache-dir", cache, "--bootstrap", "50", "--segment-duration", str(SEGMENT_SECONDS)]
                mv.main(["--variants", "base", "bal", "--configs", "VEC_tr", "--heads", "kan", "mlp", "linear", "--out", str(root / "vec"), "--checkpoints", str(root / "ckpt"),
                         "--max-epochs", "3", "--patience", "2", "--batch-size", "8", "--threads", "1"] + common)
                mv.main(["--classical", "lr_tr", "lr_tr_bal", "--out", str(root / "classical")] + common)
                mv.main(["--classical", "lr_tr", "--permute-labels", "7", "--out", str(root / "permuted")] + common)
            finally:
                os.chdir(cwd)
            with open(root / "vec" / "summary.csv", newline="") as handle:
                vec = {r["method"]: r for r in csv.DictReader(handle)}
            self.assertEqual(set(vec), {f"VEC_tr|head={h}|variant={v}" for h in ("kan", "mlp", "linear") for v in ("base", "bal")})
            with open(root / "classical" / "summary.csv", newline="") as handle:
                classical = {r["method"]: float(r["accuracy"]) for r in csv.DictReader(handle)}
            with open(root / "permuted" / "summary.csv", newline="") as handle:
                permuted = {r["method"]: float(r["accuracy"]) for r in csv.DictReader(handle)}
            self.assertGreater(classical["classical:lr_tr|descriptor=tr_stats"], 0.7)                  # the planted band is learned from whole recordings ...
            self.assertGreater(classical["classical:lr_tr|descriptor=tr_stats"], permuted["classical:lr_tr|descriptor=tr_stats"] + 0.15)   # ... and not with the labels permuted


class TestRepositoryMetadata(unittest.TestCase):
    """What a visitor meets first: the citation file, the license file and the relative links of the Markdown documents."""

    def test_citation_file_and_license_file(self):
        citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        self.assertEqual(citation["cff-version"], "1.2.0")
        self.assertEqual(citation["license"], "CC-BY-4.0")
        self.assertIn("Giacomelli", [author["family-names"] for author in citation["authors"]])
        self.assertTrue(all("family-names" in author and "given-names" in author for author in citation["authors"]))
        self.assertEqual(citation["preferred-citation"]["doi"], "10.1109/IS264627.2025.11284580")
        self.assertIn(citation["references"][0]["type"], {"data", "database"})                   # CFF 1.2.0 has no type "dataset"
        accepted = [reference for reference in citation["references"] if reference.get("status")]
        self.assertTrue(accepted and all("doi" not in reference for reference in accepted))      # accepted for presentation, no DOI yet
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("Attribution 4.0 International"))
        self.assertIn("Creative Commons Attribution 4.0 International Public License", license_text)

    def test_relative_links_of_the_markdown_documents_resolve(self):
        import re
        link = re.compile(r"\]\(([^)\s]+)\)")
        documents = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "docs" / "PROTOCOL.md"] + sorted((ROOT / "results").glob("*/README.md"))
        checked = 0
        for document in documents:
            for target in link.findall(document.read_text(encoding="utf-8")):
                if re.match(r"^(https?:|mailto:|#)", target):
                    continue
                checked += 1
                self.assertTrue((document.parent / target.split("#")[0]).exists(), f"{document.relative_to(ROOT)}: broken link {target}")
        self.assertGreater(checked, 10)


if __name__ == "__main__":
    unittest.main(verbosity=1)
