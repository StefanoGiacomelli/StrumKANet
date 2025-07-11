import os
import sys
import numpy as np
import torch
import librosa
sys.path.append(os.path.join(os.path.dirname(__file__), "rp_extract"))
from rp_extract.rp_extract import rp_extract
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


def rhythm_features(x: np.ndarray = None,
                    sr: int = 44100,
                    file_path: str = None,
                    extractors: list = ["librosa", "essentia", "rp_extract"],
                    **kwargs) -> dict:
    """
    Extract rhythmic features from an input audio array/file, using LibRoSa, Essentia and Rhythmic Patterns extractor.

    Args:
        y (np.ndarray): Audio waveform. If None, loaded from `audio_path`.
        sr (int): Sampling rate for librosa loading or override.
        audio_path (str): Path to the audio file.
        mono (bool): Convert to mono if stereo (via channel averaging).
        extractors (list): List of feature extractor to compute. Options include:
            - "librosa"
            - "essentia"
            - "rp_extract"
                * 'rp'   : np.ndarray, shape (1440,)
                    Flattened Rhythm Pattern matrix (60 mod. freq bins x 24 Bark bands).
                    Rhythm Patterns (also called Fluctuation Patterns) describe modulation amplitudes for a range of modulation frequencies 
                    on "critical bands" of the human auditory range, i.e. fluctuations (or rhythm) on a number of frequency bands. 
                    The feature extraction process for the Rhythm Patterns is composed of two stages:
                    First, the specific loudness sensation in different frequency bands is computed, by using a Short Time FFT, grouping the resulting 
                    frequency bands to psycho-acoustically motivated critical-bands, applying spreading functions to account for masking effects 
                    and successive transformation into the decibel, Phon and Sone scales. This results in a power spectrum that reflects human 
                    loudness sensation (Sonogram).
                    In the second step, the spectrum is transformed into a time-invariant representation based on the modulation frequency, 
                    which is achieved by applying another discrete Fourier transform, resulting in amplitude modulations of the loudness in 
                    individual critical bands. These amplitude modulations have different effects on human hearing sensation depending on their frequency, 
                    the most significant of which, referred to as fluctuation strength, is most intense at 4 Hz and decreasing towards 15 Hz. 
                    From that data, reoccurring patterns in the individual critical bands, resembling rhythm, are extracted, which - after applying 
                    Gaussian smoothing to diminish small variations - result in a time-invariant, comparable representation of the rhythmic patterns 
                    in the individual critical bands.

                * 'rh'   : np.ndarray, shape (60,)
                    Rhythm Histogram with 60 bins.
                    The Rhythm Histogram features we use are a descriptor for general rhythmics in an audio document. 
                    Contrary to the Rhythm Patterns and the Statistical Spectrum Descriptor, information is not stored per critical band. 
                    Rather, the magnitudes of each modulation frequency bin of all critical bands are summed up, to form a histogram of "rhythmic energy" 
                    per modulation frequency. The histogram contains 60 bins which reflect modulation frequency between 0 and 10 Hz. 
                    For a given piece of audio, the Rhythm Histogram feature set is calculated by taking the median of the histograms of every 6 second 
                    segment processed.

                * 'tssd' : np.ndarray, shape (1176,)
                    Temporal SSD statistics: 7 stats x 7 stats x 24 bands.
                    Feature sets are frequently computed on a per segment basis and do not incorporate time series aspects. 
                    As a consequence, TSSD features describe variations over time by including a temporal dimension. 
                    Statistical measures (mean, median, variance, skewness, kurtosis, min and max) are computed over the individual statistical 
                    spectrum descriptors extracted from segments at different time positions within a piece of audio. 
                    This captures timbral variations and changes over time in the audio spectrum, for all the critical Bark-bands. 
                    Thus, a change of rhythmic, instruments, voices, etc. over time is reflected by this feature set. 
                    The dimension is 7 times the dimension of an SSD (i.e. 1176).

                * 'trh'  : np.ndarray, shape (420,)
                    Temporal RH statistics: 7 stats x 60 bins.
                    Statistical measures (mean, median, variance, skewness, kurtosis, min and max) are computed over the individual 
                    Rhythm Histograms extracted from various segments in a piece of audio. 
                    Thus, change and variation of rhythmic aspects in time are captured by this descriptor.
        
        **kwargs: Additional keyword arguments routed to the individual feature extractors:
            LibRoSa:
            --------
            - onset_strength (Compute a spectral flux onset strength envelope):
                * sr (int > 0): Sampling rate of the incoming audio.
                * detrend (bool): Filter the onset strength to remove the DC component (default: False).
                * aggregate (function): Aggregation function to use when combining onsets at different frequency bins (default: np.mean)
            
            - tempogram (Compute the tempogram: local autocorrelation of the onset strength envelope):
                * sr (int > 0): Sampling rate of the incoming audio.
                * window (string, function, number, tuple, or np.ndarray [shape=(win_length,)]): A window specification as in stft.
                * win_length (int): length of the onset autocorrelation window (in frames/onset measurements) 
                                    The default settings (384) corresponds to 384 * hop_length / sr ~= 8.9s.
                * center (bool): If True, onset windows are centered. If False, windows are left-aligned.
                * hop_length (int): Number of samples between successive onset measurements.
                * norm (np.inf, -np.inf, 0, float > 0, None): Normalization mode. Set to None to disable normalization.
            
            - tempogram_ratio (Tempogram ratio features, also known as spectral rhythm patterns. This function summarizes the energy 
                               at metrically important multiples of the tempo. For example, if the tempo corresponds to the quarter-note 
                               period, the tempogram ratio will measure the energy at the eighth note, sixteenth note, half note, 
                               whole note, etc. periods, as well as dotted and triplet ratios):
                * sr (int > 0): Sampling rate of the incoming audio.
                * window (string, function, number, tuple, or np.ndarray [shape=(win_length,)]): A window specification as in stft.
                * win_length (int > 0): window length of the autocorrelation window for tempogram calculation.
                * center (bool): If True, onset windows are centered. If False, windows are left-aligned.
                * hop_length (int > 0): hop length of the time series.
                * norm (np.inf, -np.inf, 0, float > 0, None): Normalization mode. Set to None to disable normalization.
                * start_bpm (float > 0): Initial guess of the BPM if bpm is not provided.
                * std_bpm (float > 0): Standard deviation of tempo distribution.
                * max_tempo (float > 0): If provided, only estimate tempo below this threshold.

            - beat_track (Dynamic programming beat tracker. Beats are detected in three stages, following: Measure onset strength,
                          Estimate tempo from onset correlation, Pick peaks approximately consistent with estimated tempo):
                * start_bpm (float): Initial guess for the BPM (default: 120.0).
                * tightness (float): Tightness of beat distribution around tempo (default: 100).
                * trim (bool): Trim leading/trailing beats with weak onsets (default: True).
                * bpm (float): If given, override start_bpm.
                * units (str): Set to 'frames', 'samples', or 'time' (default: 'frames').
            
            Essentia:
            --------
            - rhythm_extractor (see Essentia RhythmExtractor2013 Tutorial at: 
                                https://essentia.upf.edu/tutorial_rhythm_beatdetection.html):
                * method (str): Beat tracking method, 'multifeature' or 'degara' (default: 'multifeature').
                * maxTempo (float [60, 250]): Max estimated tempo (default: 208).
                * minTempo (float [40, 180]): Min estimated tempo (default: 40).

    Returns:
        dict: Keys are feature names, values are outputs (arrays or dicts).
    """
    if x is None:
        assert file_path is not None, "Must provide either `x` or `file_path`."
        x, sr = librosa.load(file_path, sr=sr, mono=False)

    # Downmix to mono
    if x is not None:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim > 1:
            x = np.mean(x, axis=0)  # downmix stereo to mono

    # Resample to 44100 Hz
    if sr != 44100:
        x = librosa.resample(x, orig_sr=sr, target_sr=44100)
        sr = 44100

    results = {}

    if "librosa" in extractors:
        onset_env = librosa.onset.onset_strength(y=x, 
                                                 sr=sr, 
                                                 **{k: v for k, v in kwargs.items() if k in ["detrend", 
                                                                                             "aggregate"]})
        
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, 
                                              sr=sr, 
                                              **{k: v for k, v in kwargs.items() if k in ["window", 
                                                                                          "win_length", 
                                                                                          "center", 
                                                                                          "hop_length", 
                                                                                          "norm"]})
        
        tempogram_ratio = librosa.feature.tempogram_ratio(onset_envelope=onset_env, 
                                                          sr=sr, 
                                                          **{k: v for k, v in kwargs.items() if k in ["window",
                                                                                                      "win_length", 
                                                                                                      "center", 
                                                                                                      "hop_length", 
                                                                                                      "norm", 
                                                                                                      "start_bpm", 
                                                                                                      "std_bpm", 
                                                                                                      "max_tempo"]})
        
        bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, 
                                             sr=sr, 
                                             **{k: v for k, v in kwargs.items() if k in ["start_bpm",
                                                                                         "tightness", 
                                                                                         "trim", 
                                                                                         "bpm", 
                                                                                         "units"]})
        
        results["librosa"] = {"onset_strength": onset_env,          # array of onset strength values (..., m)
                              "tempogram": tempogram,               # localized autocorrelation of the onset envelope. (..., win_size, n)  
                              "tempogram_ratio": tempogram_ratio,   # tempogram ratio features (..., win_size, f=12)
                              "bpm_librosa": bpm,                           # scalar       
                              "beats": beats}                       # array of beat event locations (in frames)

    if "rp_extract" in extractors:
        features = rp_extract(wavedata                        = x,
                              samplerate                      = sr,
                              extract_rp                      = True,
                              extract_rh                      = True,
                              extract_rh2                     = True,
                              extract_mvd                     = False,
                              extract_ssd                     = False,
                              extract_trh                     = True,
                              extract_tssd                    = True,
                              spectral_masking                = True,
                              transform_db                    = True,
                              transform_phon                  = True, 
                              transform_sone                  = True, 
                              fluctuation_strength_weighting  = True, 
                              skip_leadin_fadeout             = 1,
                              step_width                      = 1)
        
        results["rp_extract"] = features

    if "essentia" in extractors:
        if not ESSENTIA_AVAILABLE:
            raise ImportError("Essentia is not installed or not available.")
        extractor = es.RhythmExtractor2013(method="multifeature", **{k: v for k, v in kwargs.items() if k in ["method",
                                                                                                              "maxTempo", 
                                                                                                              "minTempo"]})
        bpm, ticks, confidence, estimates, _ = extractor(x)
        
        results["essentia"] = {"bpm_essentia": bpm,                          # scalar
                               "ticks": ticks,                      # array of tick positions (in seconds)
                               "confidence": confidence,            # scalar
                               "bpm_estimates": estimates}              # array of bpm values characterizing distribution

    return results

#############################################################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.interpolate
    from tqdm import tqdm
    from time import perf_counter
    from globals import AUDIO_FOLDER, AUDIO_EXT, SVG_FOLDER, EXTRACTORS
    
    os.makedirs(SVG_FOLDER, exist_ok=True)

    # === List all files ===
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(AUDIO_EXT)]
    if not audio_files:
        print("No audio files found in the specified folder.")
        exit(1)

    for file in tqdm(audio_files, desc="Analyzing Rhythm features"):
        file_path = os.path.join(AUDIO_FOLDER, file)
        file_basename = os.path.splitext(file)[0]

        try:
            # Load audio file             
            x, sr = librosa.load(file_path, sr=None, mono=False)

            # Compute features
            perf_start_time = perf_counter()
            feats = rhythm_features(x=x, sr=sr, extractors=EXTRACTORS)
            perf_end_time = perf_counter()
            print(f"File: {file_basename}")
            print(f"Processed in {perf_end_time - perf_start_time:.2f}s")
          
            # LibRoSa features ---------------------------------------------------------------------------------------
            if "librosa" in feats:
                svg_path = os.path.join(SVG_FOLDER, f"{file_basename}_librosa.svg")

                onset = feats["librosa"]["onset_strength"]
                tempogram = feats["librosa"]["tempogram"]
                tempogram_ratio = feats["librosa"]["tempogram_ratio"]
                beats = feats["librosa"]["beats"]
                bpm = feats["librosa"]["bpm"]
                try:
                    bpm_val = float(bpm.item())
                except Exception:
                    bpm_val = 0.0
                
                # Print features (DEBUG)
                print(f"  Sampling Rate: {sr} Hz")
                print(f"  Onset Strength: {onset.shape}")
                print(f"  Tempogram: {tempogram.shape}")
                print(f"  Tempogram Ratio: {tempogram_ratio.shape}")
                print(f"  LibRoSa Beats: {len(beats)}")
                print(f"  LibRoSa BPM: {bpm_val:.2f}")

                # Axes & Assets
                hop_length = 512
                n_rows, n_cols = tempogram.shape
                total_duration = len(x) / sr
                center_time = total_duration / 2
                half_window = 15                                        # 30 seconds total window
                t_start = max(0, center_time - half_window)
                t_end = min(total_duration, center_time + half_window)

                time_axis = np.arange(len(x)) / sr
                onset_time_axis = librosa.frames_to_time(np.arange(len(onset)), sr=44100, hop_length=hop_length)
                frame_times = librosa.frames_to_time(np.arange(n_cols), sr=44100, hop_length=hop_length)
                beat_times = librosa.frames_to_time(beats, sr=44100, hop_length=hop_length)

                # Plot
                fig, axs = plt.subplots(5, 1, figsize=(20, 25), sharex=True)
                fig.suptitle(f"LibRoSa Rhythmic Analysis: {str(file_basename)} - {sr}Hz", fontsize=16)

                # 1. Waveform
                axs[0].plot(time_axis, x, color="steelblue")
                axs[0].set_title(f"Waveform")
                axs[0].set_ylabel("Amplitude")
                axs[0].set_xlim(t_start, t_end)
                axs[0].grid(True)

                # 2. Onset Strength
                axs[1].plot(onset_time_axis, onset / np.max(onset), color="darkgreen", label="Onset Strength")
                axs[1].vlines(beat_times, ymin=0, ymax=1.05, colors='red', linestyles='--', label="Beats")
                axs[1].set_title("Onset Strength Envelope with Beats")
                axs[1].set_ylabel("Normalized Strength")
                axs[1].set_ylim(0, 1.05)
                axs[1].set_xlim(t_start, t_end)
                axs[1].legend(loc="upper right")
                axs[1].grid(True)

                # 3. Tempogram
                bpms_linear = librosa.tempo_frequencies(n_rows, sr=sr, hop_length=hop_length)
                bpm_log_min = max(30, np.min(bpms_linear[bpms_linear > 0]))
                bpm_log_max = min(300, np.max(bpms_linear))
                bpms_log = np.geomspace(bpm_log_min, bpm_log_max, num=n_rows)
                interp_func = scipy.interpolate.interp1d(bpms_linear, tempogram, axis=0, kind='linear', bounds_error=False, fill_value=0.0)
                tempogram_log = interp_func(bpms_log)

                extent = [frame_times[0], frame_times[-1], bpms_log[0], bpms_log[-1]]
                axs[2].imshow(tempogram_log,
                            aspect='auto',
                            origin='upper',
                            cmap='magma',
                            extent=extent)
                axs[2].set_title(f"Tempogram [BPM = {bpm_val:.2f}]")
                axs[2].set_ylabel("BPM")
                axs[2].grid(True)
                axs[2].set_xlim(t_start, t_end)

                # 4. Tempogram Ratio
                ratio_labels = ["1/4", "3/8", "1/3", "1/2", "3/4", "2/3", "1", "3/2", "4/3", "2", "3", "8/3", "4"]
                axs[3].imshow(tempogram_ratio, 
                            aspect='auto', 
                            origin='lower', 
                            cmap='magma', 
                            extent=[onset_time_axis[0], onset_time_axis[-1], 0, len(ratio_labels)])
                axs[3].set_yticks(np.arange(len(ratio_labels)) + 0.5)
                axs[3].set_yticklabels(ratio_labels)
                axs[3].set_title("Tempogram Ratio")
                axs[3].set_ylabel("Rhythmic Factor")
                axs[3].set_xlim(t_start, t_end)
                axs[3].grid(True)

                # 5. Beats over waveform
                axs[4].plot(time_axis, x, color="lightgray", alpha=0.9)
                axs[4].vlines(beat_times, ymin=min(x), ymax=max(x), colors='red', linestyles='--')
                axs[4].set_title("Beats")
                axs[4].set_xlabel("Time (s)")
                axs[4].set_ylabel("Amplitude")
                axs[4].set_xlim(t_start, t_end)
                axs[4].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(svg_path)
                plt.close(fig)
            
            # RP_extract features ---------------------------------------------------------------------------------------
            if "rp_extract" in feats:
                svg_path = os.path.join(SVG_FOLDER, f"{file_basename}_rp_extract.svg")
                
                rp = feats["rp_extract"]["rp"].reshape((24, 60), order='F')
                rh = feats["rp_extract"]["rh"]
                tssd = feats["rp_extract"]["tssd"].reshape((24, 7, 7), order='F')
                trh = feats["rp_extract"]["trh"].reshape((60, 7), order='F')

                # Print features (DEBUG)
                print(f"  Rhythm Pattern: {rp.shape}")
                print(f"  Rhythm Histogram: {rh.shape}")
                print(f"  Temporal Statistical Spectrum Descriptor: {tssd.shape}")
                print(f"  Temporal Rhythm Histogram: {trh.shape}")

                # Plot
                fig, axs = plt.subplots(2, 2, figsize=(20, 20))
                axs = axs.flatten()
                fig.suptitle(f"Rhythmic Pattern Features: {str(file_basename)} - {sr}Hz", fontsize=16)

                # 1. Rhythm Pattern 
                img1 = axs[0].imshow(rp, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
                axs[0].set_title("Rhythm Patterns")
                axs[0].set_xlabel("Modulation Frequency Index")
                axs[0].set_ylabel("Frequency [Bark]")
                axs[0].grid(True)

                # 2. Rhythm Histogram
                mod_freq_res = 1.0 / (2**18 / 44100.0)  # modulation frequency resolution
                plot_index = np.arange(0, len(rh), 5)
                plot_base = plot_index + 1
                bpm_xticks = np.around(plot_base * mod_freq_res * 60).astype(int)

                axs[1].bar(np.arange(len(rh)), rh, width=0.9, color='darkgreen')
                axs[1].set_title("Rhythm Histogram (0 - 10 Hz)")
                axs[1].set_xlabel("BPM")
                axs[1].set_ylabel("Occurencies")
                axs[1].set_xticks(plot_index)
                axs[1].set_xticklabels(bpm_xticks)
                axs[1].grid(True)

                # 3. Temporal Statistical Spectrum Descriptor
                tssd_transposed = tssd.transpose(2, 1, 0)  # (Segments, Stats, Bark)
                tssd_concat = np.concatenate([seg.T for seg in tssd_transposed], axis=1)  # (Bark=24, 7×7=49)

                # Plot
                img = axs[2].imshow(tssd_concat, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
                axs[2].set_title("Temporal Statistical Spectrum Descriptor")
                axs[2].set_xlabel("Segment x Statistic")
                axs[2].set_ylabel("Frequency [Bark]")

                # Set xticks: center of each 7-wide segment group
                xticks = np.arange(7) * 7 + 3  # center of each segment block
                xtick_labels = [f"Seg {i+1}" for i in range(7)]
                axs[2].set_xticks(xticks)
                axs[2].set_xticklabels(xtick_labels)

                # Optional: vertical separators between segments
                for i in range(1, 7):
                    axs[2].axvline(i * 7 - 0.5, color='white', linestyle='--', linewidth=1)

                # Optional: Bark band ticks
                axs[2].set_yticks(np.arange(0, 25, 6))
                axs[2].set_yticklabels([str(b) for b in np.arange(0, 25, 6)])
                axs[2].grid(True)

                # 4. Temporal Rhythm Histogram Statistics
                stat_labels = ['Mean', 'Median', 'Variance', 'Skewness', 'Kurtosis', 'Min', 'Max']
                img3 = axs[3].imshow(trh, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
                axs[3].set_title("Temporal Rhythm Histogram Statistics")
                axs[3].set_xlabel("Statistics")
                axs[3].set_ylabel("Modulation Frequency Bin")
                axs[3].set_xticks(np.arange(len(stat_labels)))
                axs[3].set_xticklabels(stat_labels)
                axs[3].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(svg_path)
                plt.close(fig)
            
            # Essentia features ---------------------------------------------------------------------------------------
            if "essentia" in feats:
                essentia_path = os.path.join(SVG_FOLDER, f"{file_basename}_essentia.json")
                
                bpm = feats["essentia"]["bpm"]
                ticks = feats["essentia"]["ticks"]
                confidence = feats["essentia"]["confidence"]
                bpm_estimates = feats["essentia"]["estimates"]
                
                # Print features (DEBUG)
                print(f"  Essentia BPM: {bpm}")
                print(f"  Essentia Ticks: {len(ticks)}")
                print(f"  Confidence: {confidence}")
                print(f"  BPM Estimates: {len(bpm_estimates)}")

                # Axes & Assets
                duration = len(x) / sr
                t_center = duration / 2
                t_start = max(0, t_center - 15)
                t_end = min(duration, t_center + 15)

                start_sample = int(t_start * sr)
                end_sample = int(t_end * sr)
                x_crop = x[start_sample:end_sample]
                time_axis = np.linspace(t_start, t_end, len(x_crop))

                ticks_in_window = [t for t in ticks if t_start <= t <= t_end]

                # Plot
                fig, axs = plt.subplots(2, 1, figsize=(20,10))
                fig.suptitle(f"Essentia Rhythmic Analysis: {file_basename} - {sr}Hz", fontsize=16)

                # 1. Ticks over Waveform
                axs[0].plot(time_axis, x_crop, color="gray", alpha=0.8)
                axs[0].vlines(ticks_in_window, ymin=min(x_crop), ymax=max(x_crop), colors='red', linestyles='--', linewidth=1)
                axs[0].set_title("Beats")
                axs[0].set_xlabel("Time (s)")
                axs[0].set_ylabel("Amplitude")
                axs[0].set_xlim(t_start, t_end)
                axs[0].grid(True)

                # 2. BPMs Histogram (with predicted BPM highlighted)
                bins = np.arange(30, 301, 2)
                hist, _ = np.histogram(bpm_estimates, bins=bins)

                axs[1].bar(bins[:-1], hist, width=1.8, align='edge', color='lightgray', edgecolor='none')
                closest_bin = bins[np.abs(bins - bpm).argmin()]
                axs[1].bar(closest_bin, hist[np.abs(bins - bpm).argmin()], width=1.8, color='black', label=f"Predicted BPM: {bpm:.2f}")

                axs[1].set_title(f"BPMs Distribution (Confidence: {confidence:.2f})")
                axs[1].set_xlabel("BPM")
                axs[1].set_ylabel("Occurences")
                axs[1].legend()
                axs[1].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])

                # ---- Salvataggio ----
                save_path = os.path.join(SVG_FOLDER, f"{file_basename}_essentia.svg")
                plt.savefig(save_path)
                plt.close(fig)

            feature_out_path = os.path.join(SVG_FOLDER, f"{file_basename}_features.npz")
            np.savez_compressed(feature_out_path,
                                sr=sr,
                                onset=onset,
                                tempogram=tempogram,
                                tempogram_ratio=tempogram_ratio,
                                librosa_beats=beats,
                                librosa_bpm=bpm_val,
                                rp=rp,
                                rh=rh,
                                tssd=tssd,
                                trh=trh,
                                essentia_bpm=bpm,
                                essentia_beats=ticks,
                                essentia_confidence=confidence,
                                bpm_estimates=bpm_estimates)
            print('--------------------------------------------------------------------------------------------------------------')

        except Exception as e:
            print(f"Error processing {file}: {e}")
