//! Integration tests for the Swift→Rust FFI bindings.
//!
//! These tests exercise the FFI surface without requiring large model downloads
//! or network access. They verify:
//!   * Bridge lifecycle (create / drop / cleanup) does not crash
//!   * Pre-init availability checks return `false` instead of panicking
//!   * Strings allocated in Swift survive the round trip into Rust
//!   * Error paths (missing files) are reported instead of segfaulting
//!
//! Tests that require network downloads or compiled CoreML models are guarded
//! with `#[ignore]` and can be run via:
//!     cargo test --test ffi_bindings -- --ignored
//!
//! The diarization tests are *not* ignored: they need no download, only a staged
//! Sortformer package named by `FLUIDAUDIO_TEST_DIARIZE_MODEL` /
//! `FLUIDAUDIO_TEST_DIARIZE_AUDIO`, and skip themselves when it is absent — so a
//! machine that has one runs them by default rather than only on request.

use fluidaudio_rs::{
    DiarizeCancelToken, DiarizeComputeUnits, DiarizeEvent, DiarizeOutcome, DiarizeProgress,
    FluidAudio, FluidAudioError, KokoroComputeUnits,
};

/// The bridge can be created and dropped without panicking. Drop must not
/// crash even if no `init_*` method was ever called.
#[test]
fn bridge_create_and_drop_is_safe() {
    let audio = FluidAudio::new().expect("bridge creation should succeed");
    drop(audio);
}

/// Multiple bridges can coexist. The Swift side maintains a `globalBridge`
/// pointer for compatibility but each Rust handle owns its own retained
/// instance, so creating a second bridge must not invalidate the first.
#[test]
fn multiple_bridges_can_be_created() {
    let a = FluidAudio::new().expect("first bridge");
    let b = FluidAudio::new().expect("second bridge");

    // Both should report consistent system information; that proves both
    // pointers are still valid and the FFI calls succeed on each.
    let info_a = a.system_info();
    let info_b = b.system_info();
    assert_eq!(info_a.platform, info_b.platform);
    assert_eq!(info_a.is_apple_silicon, info_b.is_apple_silicon);
}

/// `cleanup` must be idempotent. The destructor calls it again on drop, so
/// calling it explicitly first must not double-free.
#[test]
fn cleanup_is_idempotent() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.cleanup();
    audio.cleanup();
    // Drop runs cleanup once more; should still be safe.
}

/// Before any `init_*` call, every `is_*_available` accessor must return
/// `false` rather than reading uninitialized state.
#[test]
fn availability_is_false_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(
        !audio.is_asr_available(),
        "ASR should be unavailable pre-init"
    );
    assert!(
        !audio.is_streaming_asr_available(),
        "streaming ASR should be unavailable pre-init"
    );
    assert!(
        !audio.is_vad_available(),
        "VAD should be unavailable pre-init"
    );
    assert!(
        !audio.is_diarization_available(),
        "diarization should be unavailable pre-init"
    );
}

/// `system_info()` exercises Swift→Rust string ownership: the Swift side
/// allocates with `strdup`, hands a pointer to Rust, and Rust frees it via
/// `fluidaudio_free_string`. The platform string must come back populated.
#[test]
fn system_info_round_trips_swift_strings() {
    let audio = FluidAudio::new().expect("bridge creation");
    let info = audio.system_info();

    assert!(
        info.platform == "macOS" || info.platform == "iOS",
        "unexpected platform: {}",
        info.platform
    );
    assert!(!info.chip_name.is_empty(), "chip name should be populated");
    assert!(
        info.memory_gb > 0.0,
        "memory should be reported as positive"
    );
}

/// `is_apple_silicon()` reads `SystemInfo.isAppleSilicon` from Swift and
/// must return a value consistent with `system_info()`.
#[test]
fn is_apple_silicon_matches_system_info() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert_eq!(
        audio.is_apple_silicon(),
        audio.system_info().is_apple_silicon
    );
}

/// `is_intel_mac()` and `is_apple_silicon()` must be mutually exclusive: a
/// process is either arm64 or x86_64, never both, never neither (for the
/// architectures FluidAudio supports).
#[test]
fn is_intel_mac_and_is_apple_silicon_are_exclusive() {
    let audio = FluidAudio::new().expect("bridge creation");
    let arm = audio.is_apple_silicon();
    let intel = audio.is_intel_mac();
    assert!(
        arm != intel,
        "expected exactly one of (apple_silicon, intel_mac) to be true, got both={arm} and intel={intel}"
    );
}

/// File-based methods must validate the path before crossing the FFI boundary,
/// so a missing file returns `FileNotFound` rather than panicking inside Swift.
#[test]
fn missing_file_returns_file_not_found() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .transcribe_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("transcribe_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));

    let err = audio
        .transcribe_file_streaming("/this/path/definitely/does/not/exist.wav")
        .expect_err("transcribe_file_streaming with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));

    let err = audio
        .diarize_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("diarize_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));
}

/// Calling streaming ASR session methods before initializing must surface a
/// bridge error rather than panicking. We don't assert on the message because
/// it depends on the Swift error path; we just want to confirm we get back a
/// `Result::Err` cleanly across the FFI.
#[test]
fn streaming_asr_session_methods_error_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(audio.streaming_asr_start().is_err());
    assert!(audio.streaming_asr_feed(&[0.0_f32; 1600]).is_err());
    assert!(audio.streaming_asr_finish().is_err());
}

/// VAD `process` methods must error before `init_vad` rather than crash.
#[test]
fn vad_process_errors_before_init() {
    let audio = FluidAudio::new().expect("bridge creation");
    assert!(audio.vad_process_samples(&[0.0_f32; 4096]).is_err());
    assert!(audio
        .vad_process_file("/this/path/definitely/does/not/exist.wav")
        .is_err());
}

/// VAD `process_file` validates the path on the Rust side and returns
/// `FileNotFound` (not a Swift-side error) when the path doesn't exist.
#[test]
fn vad_process_file_returns_file_not_found() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .vad_process_file("/this/path/definitely/does/not/exist.wav")
        .expect_err("vad_process_file with missing path must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));
}

/// ITN does not require model loading — `TextNormalizer.shared` is always
/// available. Calling `itn_normalize` on a fresh bridge should round-trip a
/// string from Swift back to Rust without crashing.
#[test]
fn itn_normalize_round_trips() {
    let audio = FluidAudio::new().expect("bridge creation");
    // Use a string that won't be modified by ITN regardless of which backend
    // is active. We only care that we got a String back, not that it matches
    // a specific normalized form.
    let result = audio.itn_normalize("hello").expect("itn_normalize");
    assert!(!result.is_empty(), "ITN result should not be empty");
}

/// Sentence-mode ITN must also work without init and accept arbitrary text.
#[test]
fn itn_normalize_sentence_round_trips() {
    let audio = FluidAudio::new().expect("bridge creation");
    let result = audio
        .itn_normalize_sentence("the quick brown fox")
        .expect("itn_normalize_sentence");
    assert!(!result.is_empty());

    let result = audio
        .itn_normalize_sentence_max_span("five plus five", 8)
        .expect("itn_normalize_sentence_max_span");
    assert!(!result.is_empty());
}

/// `itn_is_native_available()` must return a definite bool without crashing,
/// regardless of whether the native NeMo library is loaded in this process.
#[test]
fn itn_is_native_available_returns_bool() {
    let audio = FluidAudio::new().expect("bridge creation");
    let _ = audio.itn_is_native_available();
}

// ---------------------------------------------------------------------------
// Heavier tests below: gated behind `--ignored` because they download models
// from HuggingFace and warm up the Apple Neural Engine (cold start ~20s).
// ---------------------------------------------------------------------------

/// VAD initialization downloads a small CoreML model on first run.
#[test]
#[ignore = "downloads VAD model from HuggingFace on first run"]
fn vad_initializes() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_vad(0.85).expect("VAD init");
    assert!(audio.is_vad_available());
}

/// End-to-end VAD on a silence buffer. We don't assert classification
/// (which depends on the threshold) — only that we get one frame per 4096
/// samples and the per-frame fields look sane.
#[test]
#[ignore = "downloads VAD model from HuggingFace on first run"]
fn vad_processes_silence_buffer() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_vad(0.85).expect("VAD init");

    // 2 seconds of silence at 16kHz mono = 32_000 samples = ~7.8 chunks of 4096.
    // The Swift side pads the trailing partial chunk, so expect ceil(32000/4096) = 8.
    let samples = vec![0.0_f32; 16_000 * 2];
    let frames = audio.vad_process_samples(&samples).expect("vad process");

    assert_eq!(frames.len(), 8, "expected 8 chunks for 2s at 16kHz");
    for frame in &frames {
        assert!(frame.probability >= 0.0 && frame.probability <= 1.0);
        assert!(frame.processing_time >= 0.0);
    }
}

/// End-to-end ASR sanity check on a buffer of silence. We don't assert on the
/// transcript content, only that the call returns successfully and the
/// reported duration matches the input length.
#[test]
#[ignore = "downloads Parakeet TDT models (~600MB) and triggers ANE compilation"]
fn asr_transcribes_silence_buffer() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_asr().expect("ASR init");
    assert!(audio.is_asr_available());

    // 2 seconds of silence at 16kHz mono.
    let samples = vec![0.0_f32; 16_000 * 2];
    let result = audio.transcribe_samples(&samples).expect("transcribe");
    assert!(result.duration >= 1.9 && result.duration <= 2.1);
}

/// Regression test for the bug where `transcribe_file` / `transcribe_samples`
/// carried decoder state across calls: the TDT decoder's LSTM hidden/cell
/// tensors and `lastToken` were stored in the bridge between invocations,
/// biasing the next call's predictor with the previous call's
/// end-of-utterance state. In practice this collapsed the second and
/// subsequent transcripts to a lone "." (the previous call's terminal
/// punctuation).
///
/// We transcribe a short shipped WAV fixture twice in a row and assert:
///
///   1. The first transcript is non-empty and matches the spoken phrase
///      (sanity: the model is actually running).
///   2. The second transcript equals the first byte-for-byte.
///
/// Under the bug, (1) holds but (2) fails — the second result is some
/// degenerate suffix of the first (typically just ".").
///
/// The fixture (`tests/fixtures/hello.wav`) was generated on macOS with:
///
///     say -v Samantha --file-format=WAVE --data-format=LEI16@16000 \
///         -o tests/fixtures/hello.wav "Hello world, this is a test."
///
/// Apple's TTS output isn't subject to third-party licensing; regenerating
/// it on any macOS host produces a comparable clip.
#[test]
#[ignore = "downloads Parakeet TDT models (~600MB) and triggers ANE compilation"]
fn asr_transcribe_file_is_stateless_across_calls() {
    let audio = FluidAudio::new().expect("bridge creation");
    audio.init_asr().expect("ASR init");

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hello.wav");
    assert!(fixture.exists(), "fixture not found at {:?}", fixture);

    let first = audio.transcribe_file(&fixture).expect("first transcribe");
    let second = audio.transcribe_file(&fixture).expect("second transcribe");

    // Sanity: the model produced something for our spoken phrase. We don't
    // pin the exact transcript (model versions and TTS voices vary) — just
    // that it ran.
    let first_lower = first.text.to_lowercase();
    assert!(
        first_lower.contains("hello") || first_lower.contains("test"),
        "first transcript should reference the spoken phrase, got {:?}",
        first.text,
    );

    // The actual regression assertion.
    assert_eq!(
        first.text, second.text,
        "consecutive transcribe_file calls must yield identical output on \
         identical input; got {:?} then {:?}. Decoder state is leaking \
         between calls.",
        first.text, second.text,
    );
}

/// A bridge can be rooted at a caller-owned directory. Creation alone downloads
/// nothing, so this stays in the fast set.
#[test]
fn bridge_with_models_dir_is_created() {
    let dir = std::env::temp_dir().join("fluidaudio-rs-models-root-test");
    let audio = FluidAudio::with_models_dir(&dir).expect("bridge with a models dir");
    drop(audio);
}

/// An empty directory string is accepted and means "keep the defaults", matching
/// the null case on the Swift side.
#[test]
fn bridge_with_empty_models_dir_falls_back_to_defaults() {
    let audio = FluidAudio::with_models_dir("").expect("empty dir means defaults");
    drop(audio);
}

/// The root is honored, not merely accepted: ASR init must materialise its repo
/// folder under the supplied directory rather than the platform default.
/// Downloads ~1 GB on a cold cache.
#[test]
#[ignore]
fn models_dir_receives_the_asr_repo() {
    let root = std::env::temp_dir().join("fluidaudio-rs-models-root-e2e");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create root");

    let audio = FluidAudio::with_models_dir(&root).expect("bridge");
    audio.init_asr().expect("asr init under the supplied root");

    let repo = root.join("parakeet-tdt-0.6b-v3");
    assert!(
        repo.is_dir(),
        "expected the ASR repo under {}, found: {:?}",
        root.display(),
        std::fs::read_dir(&root).map(|d| d.flatten().map(|e| e.file_name()).collect::<Vec<_>>())
    );
}

/// The same root serves Kokoro, whose `directory` is a *base* the library appends
/// its repo folder to — unlike ASR, which takes the repo directory itself.
/// Downloads ~200 MB on a cold cache.
#[test]
#[ignore]
fn models_dir_receives_the_kokoro_repo() {
    let root = std::env::temp_dir().join("fluidaudio-rs-models-root-kokoro");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create root");

    // `af_heart` is the only voice pack the upstream ANE bundle ships; asking for another
    // 404s regardless of where the root points, which would test the bundle, not the plumbing.
    let audio = FluidAudio::with_models_dir(&root).expect("bridge");
    audio
        .init_kokoro("af_heart", "en-us")
        .expect("kokoro init under the supplied root");

    let repo = root.join("kokoro-82m-coreml").join("ANE");
    assert!(
        repo.is_dir(),
        "expected the Kokoro repo under {}, found: {:?}",
        root.display(),
        std::fs::read_dir(&root).map(|d| d.flatten().map(|e| e.file_name()).collect::<Vec<_>>())
    );
}

/// Kokoro synthesises with the Neural Engine excluded. This is the path a host
/// without an ANE must take: the default mapping pins the vocoder stage to the
/// ANE, and CoreML cannot prepare that program where none is exposed — the load
/// still succeeds, so the failure only shows up here, at the first prediction.
///
/// Unlike the unit tests in `src/lib.rs`, this one actually crosses the FFI
/// boundary and runs the Swift parser, so it is the only thing that would catch
/// `as_str` drifting away from `TtsComputeUnitPreset(cliValue:)`.
///
/// Uses the default cache; downloads ~200 MB if Kokoro was never fetched.
#[test]
#[ignore]
fn kokoro_synthesises_without_the_neural_engine() {
    let audio = FluidAudio::new().expect("bridge");
    audio
        .init_kokoro_with_compute_units("af_heart", "en-us", KokoroComputeUnits::CpuAndGpu)
        .expect("kokoro init on cpu+gpu");

    let wav = audio
        .synthesize_kokoro("The quick brown fox.", "af_heart", 1.0)
        .expect("synthesis on cpu+gpu");
    assert!(
        wav.len() > 44,
        "expected audio beyond a bare WAV header, got {} bytes",
        wav.len()
    );
}

/// Synthesizing before `init_kokoro` must come back classified, not as a blanket
/// failure: a caller distinguishes a call-order bug from a missing asset by the
/// variant alone (the detail only reaches stderr). Needs no models — the guard
/// runs before any Swift model work.
#[test]
fn samples_synthesis_without_kokoro_is_classified_not_initialized() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .synthesize_kokoro_samples("Hello", "af_heart", 1.0)
        .expect_err("synthesis before init must fail");
    assert!(
        matches!(err, FluidAudioError::NotInitialized(_)),
        "unexpected error variant: {err:?}"
    );
}

/// The reason this entry point exists: `synthesize_kokoro` hands back a WAV that
/// `KokoroAneManager.wavData` peak-normalized to 0 dBFS for every variant but
/// Japanese, and the scale factor is gone by then. The samples path must be the
/// same audio *before* that slam — same duration, but not pinned to full scale.
///
/// Runs on CPU+GPU so it works where no ANE is exposed; downloads ~200 MB if
/// Kokoro was never fetched.
#[test]
#[ignore]
fn samples_synthesis_is_not_peak_normalized() {
    let audio = FluidAudio::new().expect("bridge");
    audio
        .init_kokoro_with_compute_units("af_heart", "en-us", KokoroComputeUnits::CpuAndGpu)
        .expect("kokoro init on cpu+gpu");

    let (samples, sample_rate) = audio
        .synthesize_kokoro_samples("The quick brown fox.", "af_heart", 1.0)
        .expect("sample synthesis on cpu+gpu");
    assert_eq!(sample_rate, 24_000);
    assert!(!samples.is_empty(), "expected audio, got no samples");

    let peak = samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
    assert!(
        peak > 0.0 && (peak - 1.0).abs() > 1e-3,
        "English samples peaked at {peak}, which is the 0 dBFS slam this path exists to avoid"
    );

    // Same synthesis, so the WAV must carry the same number of frames — that is
    // what proves the two paths differ only in level.
    let wav = audio
        .synthesize_kokoro("The quick brown fox.", "af_heart", 1.0)
        .expect("wav synthesis on cpu+gpu");
    let wav_frames = (wav.len() - 44) / 2;
    assert_eq!(
        wav_frames,
        samples.len(),
        "sample and WAV paths disagree on length"
    );
}

/// The English lexicon lives on the Kokoro manager, so installing one before
/// `init_kokoro` must fail rather than quietly drop the caller's pronunciations.
/// Needs no models — the guard runs before any Swift model work.
#[test]
fn english_lexicon_without_kokoro_is_an_error() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .set_kokoro_english_lexicon(&[("JSON", "ˈdʒeɪsən")])
        .expect_err("lexicon before init must fail");
    assert!(
        matches!(err, FluidAudioError::NotInitialized(_)),
        "unexpected error variant: {err:?}"
    );
}

/// A custom pronunciation reaches the synthesizer: the same word renders
/// differently once its IPA is overridden. Runs on CPU+GPU so it works where no
/// ANE is exposed; downloads ~200 MB if Kokoro was never fetched.
#[test]
#[ignore]
fn english_lexicon_changes_the_synthesized_audio() {
    let audio = FluidAudio::new().expect("bridge");
    audio
        .init_kokoro_with_compute_units("af_heart", "en-us", KokoroComputeUnits::CpuAndGpu)
        .expect("kokoro init on cpu+gpu");

    let plain = audio
        .synthesize_kokoro("JSON", "af_heart", 1.0)
        .expect("synthesis without an override");

    audio
        .set_kokoro_english_lexicon(&[("JSON", "ˈdʒeɪsən")])
        .expect("install the override");
    let overridden = audio
        .synthesize_kokoro("JSON", "af_heart", 1.0)
        .expect("synthesis with an override");

    assert_ne!(
        plain, overridden,
        "the custom lexicon did not reach the English G2P"
    );
}

/// The controlled diarize entry point validates paths on the Rust side, like the
/// plain one, so a bad path never reaches Swift.
#[test]
fn controlled_diarize_returns_file_not_found() {
    let audio = FluidAudio::new().expect("bridge creation");
    let err = audio
        .diarize_file_with_models_controlled(
            "/this/path/definitely/does/not/exist.wav",
            "/this/path/definitely/does/not/exist.mlpackage",
            DiarizeComputeUnits::All,
            None,
            None,
        )
        .expect_err("controlled diarize with missing audio must error");
    assert!(matches!(err, FluidAudioError::FileNotFound(_)));
}

/// A cancel requested before the call starts must be honoured — the token is
/// sticky, so the race between "start diarizing" and "stop" cannot be lost.
///
/// Runs against the caller's staged Sortformer package via
/// `FLUIDAUDIO_TEST_DIARIZE_MODEL`; without it there is nothing to diarize.
#[test]
fn diarize_honours_a_cancel_requested_before_the_call() {
    let (audio_path, model_path) = match diarize_test_inputs() {
        Some(paths) => paths,
        None => return,
    };
    let audio = FluidAudio::new().expect("bridge");
    let token = DiarizeCancelToken::new();
    token.cancel();

    let outcome = audio
        .diarize_file_with_models_controlled(
            &audio_path,
            &model_path,
            DiarizeComputeUnits::All,
            Some(&token),
            None,
        )
        .expect("a cancelled diarize is not an error");
    assert!(matches!(outcome, DiarizeOutcome::Cancelled));
}

/// The model-ready marker fires exactly once, before any chunk: that ordering is what
/// lets a caller bound the model load and the audio processing separately.
#[test]
fn diarize_signals_model_ready_before_the_first_chunk() {
    let (audio_path, model_path) = match diarize_test_inputs() {
        Some(paths) => paths,
        None => return,
    };
    let audio = FluidAudio::new().expect("bridge");
    let mut events: Vec<DiarizeEvent> = Vec::new();
    let mut record = |e: DiarizeEvent| events.push(e);

    let outcome = audio
        .diarize_file_with_models_controlled(
            &audio_path,
            &model_path,
            DiarizeComputeUnits::All,
            None,
            Some(&mut record),
        )
        .expect("diarize");

    assert!(matches!(outcome, DiarizeOutcome::Completed(_)));
    let ready_positions: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, DiarizeEvent::ModelReady))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        ready_positions,
        vec![0],
        "ModelReady must arrive exactly once and first, got {} events",
        events.len()
    );
}

/// Progress fires per chunk and advances monotonically to the full sample count.
#[test]
fn diarize_reports_monotonic_progress() {
    let (audio_path, model_path) = match diarize_test_inputs() {
        Some(paths) => paths,
        None => return,
    };
    let audio = FluidAudio::new().expect("bridge");
    let mut ticks: Vec<DiarizeProgress> = Vec::new();
    let mut record = |e: DiarizeEvent| {
        if let DiarizeEvent::Progress(p) = e {
            ticks.push(p);
        }
    };

    let outcome = audio
        .diarize_file_with_models_controlled(
            &audio_path,
            &model_path,
            DiarizeComputeUnits::All,
            None,
            Some(&mut record),
        )
        .expect("diarize");

    assert!(matches!(outcome, DiarizeOutcome::Completed(_)));
    assert!(!ticks.is_empty(), "expected at least one progress report");
    assert!(
        ticks
            .windows(2)
            .all(|w| w[0].chunks < w[1].chunks && w[0].processed_samples <= w[1].processed_samples),
        "progress must advance monotonically: {ticks:?}"
    );
    // Progress counts whole chunks, so a partial trailing chunk leaves the last
    // report short of the total (measured: 1466880/1478762, 99.2%). Callers must
    // not treat "processed == total" as the completion signal.
    let last = ticks.last().expect("checked non-empty");
    let ratio = last.processed_samples as f64 / last.total_samples as f64;
    assert!(
        (0.98..=1.0).contains(&ratio),
        "the final report should cover nearly all samples, got {last:?}"
    );
}

/// Cancelling mid-run stops within a chunk instead of running to completion.
#[test]
fn diarize_cancels_mid_run() {
    let (audio_path, model_path) = match diarize_test_inputs() {
        Some(paths) => paths,
        None => return,
    };
    let audio = FluidAudio::new().expect("bridge");
    let token = std::sync::Arc::new(DiarizeCancelToken::new());
    let trip = std::sync::Arc::clone(&token);
    let mut on_event = move |e: DiarizeEvent| {
        if let DiarizeEvent::Progress(p) = e {
            if p.chunks >= 3 {
                trip.cancel();
            }
        }
    };

    let outcome = audio
        .diarize_file_with_models_controlled(
            &audio_path,
            &model_path,
            DiarizeComputeUnits::All,
            Some(&token),
            Some(&mut on_event),
        )
        .expect("a cancelled diarize is not an error");
    assert!(matches!(outcome, DiarizeOutcome::Cancelled));
}

/// Audio and model paths for the diarization tests, or `None` when the caller
/// has not staged a Sortformer package to test against.
fn diarize_test_inputs() -> Option<(String, String)> {
    let model = std::env::var("FLUIDAUDIO_TEST_DIARIZE_MODEL").ok()?;
    let audio = std::env::var("FLUIDAUDIO_TEST_DIARIZE_AUDIO").ok()?;
    Some((audio, model))
}
