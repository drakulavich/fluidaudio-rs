//! Swift bridge definitions for FluidAudio bindings
//!
//! Using manual FFI instead of swift-bridge to avoid complexity with Vec types.

// Raw FFI functions - called directly from Rust, implemented in Swift
#[link(name = "FluidAudioBridge")]
extern "C" {
    // Offline enforcement (process-global; upstream's flag is a static)
    fn fluidaudio_set_offline_mode(enabled: i32);
    fn fluidaudio_offline_mode() -> i32;
    fn fluidaudio_model_registry_base_url() -> *mut i8;

    // Constructor / Destructor
    fn fluidaudio_bridge_create() -> *mut std::ffi::c_void;
    fn fluidaudio_bridge_create_with_models_dir(dir: *const i8) -> *mut std::ffi::c_void;
    fn fluidaudio_bridge_destroy(bridge: *mut std::ffi::c_void);

    // ASR
    fn fluidaudio_initialize_asr(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_transcribe_file(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;
    fn fluidaudio_is_asr_available(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_transcribe_samples(
        bridge: *mut std::ffi::c_void,
        samples: *const f32,
        sample_count: u32,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;

    // Streaming ASR
    fn fluidaudio_initialize_streaming_asr(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_streaming_asr_start(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_streaming_asr_feed(
        bridge: *mut std::ffi::c_void,
        samples: *const f32,
        count: u32,
    ) -> i32;
    fn fluidaudio_streaming_asr_finish(
        bridge: *mut std::ffi::c_void,
        out_text: *mut *mut i8,
    ) -> i32;
    fn fluidaudio_transcribe_file_streaming(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        out_text: *mut *mut i8,
        out_confidence: *mut f32,
        out_duration: *mut f64,
        out_processing_time: *mut f64,
        out_rtfx: *mut f32,
    ) -> i32;
    fn fluidaudio_is_streaming_asr_available(bridge: *mut std::ffi::c_void) -> i32;

    // VAD
    fn fluidaudio_initialize_vad(bridge: *mut std::ffi::c_void, threshold: f32) -> i32;
    fn fluidaudio_is_vad_available(bridge: *mut std::ffi::c_void) -> i32;

    // Diarization
    fn fluidaudio_initialize_diarization(bridge: *mut std::ffi::c_void, threshold: f64) -> i32;
    fn fluidaudio_diarize_file(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        out_speaker_ids: *mut *mut *mut i8,
        out_start_times: *mut *mut f32,
        out_end_times: *mut *mut f32,
        out_quality_scores: *mut *mut f32,
        out_count: *mut u32,
    ) -> i32;
    fn fluidaudio_is_diarization_available(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_free_diarization_result(
        speaker_ids: *mut *mut i8,
        start_times: *mut f32,
        end_times: *mut f32,
        quality_scores: *mut f32,
        count: u32,
    );

    // System Info
    fn fluidaudio_get_platform(out: *mut *mut i8);
    fn fluidaudio_get_chip_name(out: *mut *mut i8);
    fn fluidaudio_get_memory_gb() -> f64;
    fn fluidaudio_is_apple_silicon() -> i32;
    fn fluidaudio_is_intel_mac() -> i32;

    // VAD processing
    fn fluidaudio_vad_process_file(
        bridge: *mut std::ffi::c_void,
        path: *const i8,
        out_probabilities: *mut *mut f32,
        out_is_voice_active: *mut *mut u8,
        out_processing_times: *mut *mut f64,
        out_count: *mut u32,
    ) -> i32;
    fn fluidaudio_vad_process_samples(
        bridge: *mut std::ffi::c_void,
        samples: *const f32,
        count: u32,
        out_probabilities: *mut *mut f32,
        out_is_voice_active: *mut *mut u8,
        out_processing_times: *mut *mut f64,
        out_count: *mut u32,
    ) -> i32;
    fn fluidaudio_free_vad_result(
        probabilities: *mut f32,
        is_voice_active: *mut u8,
        processing_times: *mut f64,
        count: u32,
    );

    // ITN (Inverse Text Normalization)
    fn fluidaudio_itn_normalize(
        bridge: *mut std::ffi::c_void,
        text: *const i8,
        out_text: *mut *mut i8,
    ) -> i32;
    fn fluidaudio_itn_normalize_sentence(
        bridge: *mut std::ffi::c_void,
        text: *const i8,
        out_text: *mut *mut i8,
    ) -> i32;
    fn fluidaudio_itn_normalize_sentence_max_span(
        bridge: *mut std::ffi::c_void,
        text: *const i8,
        max_span_tokens: u32,
        out_text: *mut *mut i8,
    ) -> i32;
    fn fluidaudio_itn_is_native_available(bridge: *mut std::ffi::c_void) -> i32;

    // Cleanup
    fn fluidaudio_cleanup(bridge: *mut std::ffi::c_void);

    // String free
    fn fluidaudio_free_string(s: *mut i8);

    // Kokoro TTS
    fn fluidaudio_initialize_kokoro(
        bridge: *mut std::ffi::c_void,
        default_voice: *const i8,
        lang: *const i8,
    ) -> i32;
    fn fluidaudio_initialize_kokoro_with_compute_units(
        bridge: *mut std::ffi::c_void,
        default_voice: *const i8,
        lang: *const i8,
        compute_units: *const i8,
    ) -> i32;
    fn fluidaudio_kokoro_synthesize(
        bridge: *mut std::ffi::c_void,
        text: *const i8,
        voice: *const i8,
        speed: f32,
        out_bytes: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;
    fn fluidaudio_kokoro_free_bytes(p: *mut u8);
    fn fluidaudio_is_kokoro_available(bridge: *mut std::ffi::c_void) -> i32;
    fn fluidaudio_kokoro_set_english_lexicon(
        bridge: *mut std::ffi::c_void,
        words: *const *const i8,
        phonemes: *const *const i8,
        count: usize,
    ) -> i32;

    // Model-path diarization (loads from a pre-staged .mlpackage, no download)
    fn fluidaudio_diarize_file_with_models(
        bridge: *mut std::ffi::c_void,
        audio_path: *const i8,
        model_path: *const i8,
        out_speaker_ids: *mut *mut *mut i8,
        out_start_times: *mut *mut f32,
        out_end_times: *mut *mut f32,
        out_quality_scores: *mut *mut f32,
        out_count: *mut u32,
    ) -> i32;

    fn fluidaudio_diarize_file_with_models_controlled(
        bridge: *mut std::ffi::c_void,
        audio_path: *const i8,
        model_path: *const i8,
        compute_units: *const i8,
        cancel_token: *mut std::ffi::c_void,
        model_ready: Option<DiarizeModelReadyTrampoline>,
        progress: Option<DiarizeProgressTrampoline>,
        callback_context: *mut std::ffi::c_void,
        out_speaker_ids: *mut *mut *mut i8,
        out_start_times: *mut *mut f32,
        out_end_times: *mut *mut f32,
        out_quality_scores: *mut *mut f32,
        out_count: *mut u32,
    ) -> i32;

    fn fluidaudio_diarize_cancel_token_new() -> *mut std::ffi::c_void;
    fn fluidaudio_diarize_cancel(token: *mut std::ffi::c_void);
    fn fluidaudio_diarize_cancel_token_free(token: *mut std::ffi::c_void);

    // Pre-compile the diarization .mlpackage into a writable per-user cache and
    // load it once (warm-up; populates the CoreML ANE/e5rt cache).
    fn fluidaudio_compile_diarization_model(
        bridge: *mut std::ffi::c_void,
        model_path: *const i8,
    ) -> i32;
}

use std::ffi::{CStr, CString};

/// Swift-facing shape of the diarization progress callback.
pub type DiarizeProgressTrampoline =
    extern "C" fn(context: *mut std::ffi::c_void, processed: u64, total: u64, chunks: u32);

/// Swift-facing shape of the one-shot model-ready callback.
pub type DiarizeModelReadyTrampoline = extern "C" fn(context: *mut std::ffi::c_void);

/// One diarization progress report: `processed_samples` of `total_samples` consumed
/// after `chunks` model invocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiarizeProgress {
    pub processed_samples: u64,
    pub total_samples: u64,
    pub chunks: u32,
}

/// What a controlled diarization reports while it runs.
///
/// [`Self::ModelReady`] arrives exactly once and separates the two costs a caller
/// cannot otherwise tell apart: the model load ahead of it is fixed (cold, the ~105 s
/// ANE compile), the audio read, resample and chunking after it scale with the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiarizeEvent {
    ModelReady,
    Progress(DiarizeProgress),
}

/// Handle for stopping an in-flight [`FluidAudioBridge::diarize_file_with_models_controlled`].
///
/// Cancelling is sticky and idempotent: it may be requested before the call starts,
/// after it has returned, or several times. Only the chunk loop is interruptible, so a
/// cancel landing during the model load takes effect when the load finishes.
pub struct DiarizeCancelToken {
    ptr: *mut std::ffi::c_void,
}

// The Swift token serializes every mutation behind its own lock; that is the whole
// point of it, since one thread cancels while another is parked inside the diarize call.
unsafe impl Send for DiarizeCancelToken {}
unsafe impl Sync for DiarizeCancelToken {}

impl DiarizeCancelToken {
    pub fn new() -> Self {
        Self {
            ptr: unsafe { fluidaudio_diarize_cancel_token_new() },
        }
    }

    pub fn cancel(&self) {
        unsafe { fluidaudio_diarize_cancel(self.ptr) }
    }

    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

impl Default for DiarizeCancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DiarizeCancelToken {
    fn drop(&mut self) {
        unsafe { fluidaudio_diarize_cancel_token_free(self.ptr) }
    }
}

extern "C" fn diarize_progress_trampoline(
    context: *mut std::ffi::c_void,
    processed: u64,
    total: u64,
    chunks: u32,
) {
    // SAFETY: see `deliver_diarize_event`.
    unsafe {
        deliver_diarize_event(
            context,
            DiarizeEvent::Progress(DiarizeProgress {
                processed_samples: processed,
                total_samples: total,
                chunks,
            }),
        )
    }
}

extern "C" fn diarize_model_ready_trampoline(context: *mut std::ffi::c_void) {
    // SAFETY: see `deliver_diarize_event`.
    unsafe { deliver_diarize_event(context, DiarizeEvent::ModelReady) }
}

/// # Safety
///
/// `context` must be the `&mut &mut dyn FnMut(DiarizeEvent)` parked on the caller's
/// stack for the whole (synchronous) diarize call, or null. Swift invokes both
/// trampolines from exactly one thread at a time — the model-ready marker fires before
/// any chunk, and the diarizer holds its own lock across the chunk loop.
unsafe fn deliver_diarize_event(context: *mut std::ffi::c_void, event: DiarizeEvent) {
    if context.is_null() {
        return;
    }
    let callback = unsafe { &mut *(context as *mut &mut (dyn FnMut(DiarizeEvent) + Send)) };
    callback(event);
}

/// Outcome of a cancellable diarization.
#[derive(Debug)]
pub enum DiarizeOutcome {
    Completed(Vec<DiarizationSegment>),
    Cancelled,
}

/// Turn FluidAudio's offline-only enforcement on or off. Process-global —
/// upstream's `ModelHub.offlineMode` is a static, and it is read per request,
/// so this must be set before any loader is touched.
pub fn set_offline_mode(enabled: bool) {
    unsafe { fluidaudio_set_offline_mode(i32::from(enabled)) }
}

/// Current state of the process-global offline flag.
pub fn offline_mode() -> bool {
    unsafe { fluidaudio_offline_mode() != 0 }
}

/// The registry base every FluidAudio download URL is built from.
pub fn model_registry_base_url() -> String {
    // SAFETY: Swift hands back a `strdup`'d NUL-terminated string, or null on
    // allocation failure; ownership transfers here and is released below.
    unsafe {
        let raw = fluidaudio_model_registry_base_url();
        if raw.is_null() {
            return String::new();
        }
        let owned = CStr::from_ptr(raw).to_string_lossy().into_owned();
        fluidaudio_free_string(raw);
        owned
    }
}

/// Why a Kokoro entry point failed. The Swift side returns a status code
/// (`KokoroStatus`) rather than a message, so this is the whole of what
/// crosses the boundary; the detail is on stderr.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KokoroFailure {
    /// The engine was not initialized (call `initialize_kokoro` first).
    NotInitialized,
    /// A model asset was missing and offline mode blocked fetching it.
    AssetsUnavailable,
    /// Anything else; see the `Kokoro …: <error>` line on stderr.
    Failed,
}

impl KokoroFailure {
    fn from_status(status: i32) -> Self {
        match status {
            -2 => Self::NotInitialized,
            -3 => Self::AssetsUnavailable,
            _ => Self::Failed,
        }
    }
}

/// A failed Kokoro call: the classified reason plus the operation that hit it.
#[derive(Debug, Clone)]
pub struct KokoroError {
    pub failure: KokoroFailure,
    pub message: String,
}

impl KokoroError {
    fn new(status: i32, operation: &str) -> Self {
        let failure = KokoroFailure::from_status(status);
        let message = match failure {
            KokoroFailure::NotInitialized => format!("{operation}: Kokoro is not initialized"),
            KokoroFailure::AssetsUnavailable => {
                format!("{operation}: model assets are missing and downloads are disabled")
            }
            KokoroFailure::Failed => format!("{operation} failed"),
        };
        Self { failure, message }
    }

    fn detailed(status: i32, operation: &str, detail: String) -> Self {
        let mut err = Self::new(status, operation);
        err.message = format!("{}: {detail}", err.message);
        err
    }
}

/// Safe wrapper for the FluidAudio bridge
pub struct FluidAudioBridge {
    ptr: *mut std::ffi::c_void,
}

// The Swift bridge is thread-safe as it uses internal synchronization
unsafe impl Send for FluidAudioBridge {}
unsafe impl Sync for FluidAudioBridge {}

impl FluidAudioBridge {
    pub fn new() -> Option<Self> {
        let ptr = unsafe { fluidaudio_bridge_create() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Roots every model download at `dir` instead of FluidAudio's platform defaults.
    pub fn new_with_models_dir(dir: &str) -> Option<Self> {
        let c_dir = std::ffi::CString::new(dir).ok()?;
        let ptr = unsafe { fluidaudio_bridge_create_with_models_dir(c_dir.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn initialize_asr(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_asr(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize ASR".to_string())
        }
    }

    pub fn initialize_kokoro(&self, default_voice: &str, lang: &str) -> Result<(), KokoroError> {
        let invalid = |what: &str| KokoroError::detailed(-1, "initialize Kokoro", what.to_string());
        let c_voice = CString::new(default_voice).map_err(|_| invalid("invalid voice"))?;
        let c_lang = CString::new(lang).map_err(|_| invalid("invalid lang"))?;
        let result =
            unsafe { fluidaudio_initialize_kokoro(self.ptr, c_voice.as_ptr(), c_lang.as_ptr()) };
        if result == 0 {
            Ok(())
        } else {
            Err(KokoroError::new(result, "initialize Kokoro"))
        }
    }

    pub fn initialize_kokoro_with_compute_units(
        &self,
        default_voice: &str,
        lang: &str,
        compute_units: &str,
    ) -> Result<(), KokoroError> {
        let invalid = |what: &str| KokoroError::detailed(-1, "initialize Kokoro", what.to_string());
        let c_voice = CString::new(default_voice).map_err(|_| invalid("invalid voice"))?;
        let c_lang = CString::new(lang).map_err(|_| invalid("invalid lang"))?;
        let c_units = CString::new(compute_units).map_err(|_| invalid("invalid compute units"))?;
        let result = unsafe {
            fluidaudio_initialize_kokoro_with_compute_units(
                self.ptr,
                c_voice.as_ptr(),
                c_lang.as_ptr(),
                c_units.as_ptr(),
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(KokoroError::detailed(
                result,
                "initialize Kokoro",
                format!("compute units: {compute_units}"),
            ))
        }
    }

    /// Synthesize `text` with `voice` at `speed`; returns the complete WAV bytes
    /// produced by FluidAudio's KokoroAneManager (24 kHz mono 16-bit PCM (i16), peak-normalized).
    pub fn kokoro_synthesize(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<u8>, KokoroError> {
        let invalid = |what: &str| KokoroError::detailed(-1, "synthesize", what.to_string());
        let c_text = CString::new(text).map_err(|_| invalid("invalid text"))?;
        let c_voice = CString::new(voice).map_err(|_| invalid("invalid voice"))?;
        let mut out_bytes: *mut u8 = std::ptr::null_mut();
        let mut out_len: usize = 0;

        let result = unsafe {
            fluidaudio_kokoro_synthesize(
                self.ptr,
                c_text.as_ptr(),
                c_voice.as_ptr(),
                speed,
                &mut out_bytes,
                &mut out_len,
            )
        };

        if result != 0 {
            return Err(KokoroError::new(result, "synthesize"));
        }
        if out_bytes.is_null() || out_len == 0 {
            // On a success return Swift may still have handed us a (possibly
            // zero-length) allocation; free it (null-safe) so the empty-audio
            // path can't leak.
            unsafe { fluidaudio_kokoro_free_bytes(out_bytes) };
            return Err(KokoroError::detailed(
                -1,
                "synthesize",
                "no audio".to_string(),
            ));
        }

        // SAFETY: the Swift side allocated `out_len` bytes at `out_bytes`; copy
        // them out, then hand the buffer back to Swift to free.
        let wav = unsafe { std::slice::from_raw_parts(out_bytes, out_len) }.to_vec();
        unsafe { fluidaudio_kokoro_free_bytes(out_bytes) };
        Ok(wav)
    }

    pub fn is_kokoro_available(&self) -> bool {
        unsafe { fluidaudio_is_kokoro_available(self.ptr) != 0 }
    }

    /// Install (or clear) English pronunciation overrides on the initialized
    /// Kokoro engine. An empty slice clears the table.
    pub fn set_kokoro_english_lexicon(&self, entries: &[(&str, &str)]) -> Result<(), KokoroError> {
        let invalid = |what: String| KokoroError::detailed(-1, "set the English lexicon", what);
        let mut words = Vec::with_capacity(entries.len());
        let mut phonemes = Vec::with_capacity(entries.len());
        for (word, ipa) in entries {
            words.push(CString::new(*word).map_err(|_| invalid(format!("invalid word '{word}'")))?);
            phonemes.push(
                CString::new(*ipa)
                    .map_err(|_| invalid(format!("invalid phonemes for '{word}'")))?,
            );
        }
        let word_ptrs: Vec<*const i8> = words.iter().map(|s| s.as_ptr()).collect();
        let phoneme_ptrs: Vec<*const i8> = phonemes.iter().map(|s| s.as_ptr()).collect();

        // SAFETY: both arrays hold `entries.len()` live pointers into `words` /
        // `phonemes`, which outlive the call; Swift copies the strings.
        let result = unsafe {
            fluidaudio_kokoro_set_english_lexicon(
                self.ptr,
                word_ptrs.as_ptr(),
                phoneme_ptrs.as_ptr(),
                entries.len(),
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(KokoroError::new(result, "set the English lexicon"))
        }
    }

    /// Pre-compile the diarization `.mlpackage` to its stable `.mlmodelc` sibling
    /// and load it once (warm-up). Populates the CoreML ANE/e5rt cache so the first
    /// real diarize is fast. No audio is processed.
    pub fn compile_diarization_model(&self, model_path: &str) -> Result<(), String> {
        let c_model = CString::new(model_path).map_err(|_| "Invalid model path")?;
        let result = unsafe { fluidaudio_compile_diarization_model(self.ptr, c_model.as_ptr()) };
        if result != 0 {
            return Err("Diarization model warm-up (compile) failed".to_string());
        }
        Ok(())
    }

    /// Diarize from a pre-staged Sortformer `.mlpackage` — no network download.
    /// Marshalling mirrors `diarize_file`.
    pub fn diarize_file_with_models(
        &self,
        audio_path: &str,
        model_path: &str,
    ) -> Result<Vec<DiarizationSegment>, String> {
        let c_audio = CString::new(audio_path).map_err(|_| "Invalid audio path")?;
        let c_model = CString::new(model_path).map_err(|_| "Invalid model path")?;

        let mut speaker_ids_ptr: *mut *mut i8 = std::ptr::null_mut();
        let mut start_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut end_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut quality_scores_ptr: *mut f32 = std::ptr::null_mut();
        let mut count: u32 = 0;

        let result = unsafe {
            fluidaudio_diarize_file_with_models(
                self.ptr,
                c_audio.as_ptr(),
                c_model.as_ptr(),
                &mut speaker_ids_ptr,
                &mut start_times_ptr,
                &mut end_times_ptr,
                &mut quality_scores_ptr,
                &mut count,
            )
        };

        if result != 0 {
            return Err("Diarization (model path) failed".to_string());
        }

        Ok(unsafe {
            collect_diarization_segments(
                speaker_ids_ptr,
                start_times_ptr,
                end_times_ptr,
                quality_scores_ptr,
                count,
            )
        })
    }

    /// `diarize_file_with_models` plus progress, cancellation and compute-unit choice.
    /// `observer` is called from the diarizer's thread while this one is parked.
    pub fn diarize_file_with_models_controlled(
        &self,
        audio_path: &str,
        model_path: &str,
        compute_units: &str,
        cancel: Option<&DiarizeCancelToken>,
        observer: Option<&mut (dyn FnMut(DiarizeEvent) + Send)>,
    ) -> Result<DiarizeOutcome, String> {
        let c_audio = CString::new(audio_path).map_err(|_| "Invalid audio path")?;
        let c_model = CString::new(model_path).map_err(|_| "Invalid model path")?;
        let c_units = CString::new(compute_units).map_err(|_| "Invalid compute units")?;

        let mut speaker_ids_ptr: *mut *mut i8 = std::ptr::null_mut();
        let mut start_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut end_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut quality_scores_ptr: *mut f32 = std::ptr::null_mut();
        let mut count: u32 = 0;

        let mut callback = observer;
        let (ready, progress, context) = match callback.as_mut() {
            Some(cb) => (
                Some(diarize_model_ready_trampoline as DiarizeModelReadyTrampoline),
                Some(diarize_progress_trampoline as DiarizeProgressTrampoline),
                cb as *mut &mut (dyn FnMut(DiarizeEvent) + Send) as *mut std::ffi::c_void,
            ),
            None => (None, None, std::ptr::null_mut()),
        };

        let result = unsafe {
            fluidaudio_diarize_file_with_models_controlled(
                self.ptr,
                c_audio.as_ptr(),
                c_model.as_ptr(),
                c_units.as_ptr(),
                cancel.map_or(std::ptr::null_mut(), DiarizeCancelToken::as_ptr),
                ready,
                progress,
                context,
                &mut speaker_ids_ptr,
                &mut start_times_ptr,
                &mut end_times_ptr,
                &mut quality_scores_ptr,
                &mut count,
            )
        };

        match result {
            0 => Ok(DiarizeOutcome::Completed(unsafe {
                collect_diarization_segments(
                    speaker_ids_ptr,
                    start_times_ptr,
                    end_times_ptr,
                    quality_scores_ptr,
                    count,
                )
            })),
            -2 => Ok(DiarizeOutcome::Cancelled),
            _ => Err(format!(
                "Diarization (model path) failed (compute units: {compute_units})"
            )),
        }
    }

    pub fn transcribe_file(&self, path: &str) -> Result<AsrResult, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;

        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_transcribe_file(
                self.ptr,
                c_path.as_ptr(),
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<AsrResult, String> {
        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_transcribe_samples(
                self.ptr,
                samples.as_ptr(),
                samples.len() as u32,
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn is_asr_available(&self) -> bool {
        unsafe { fluidaudio_is_asr_available(self.ptr) != 0 }
    }

    pub fn initialize_streaming_asr(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_streaming_asr(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize streaming ASR".to_string())
        }
    }

    pub fn streaming_asr_start(&self) -> Result<(), String> {
        let result = unsafe { fluidaudio_streaming_asr_start(self.ptr) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to start streaming ASR session".to_string())
        }
    }

    pub fn streaming_asr_feed(&self, samples: &[f32]) -> Result<(), String> {
        let result = unsafe {
            fluidaudio_streaming_asr_feed(self.ptr, samples.as_ptr(), samples.len() as u32)
        };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to feed samples to streaming ASR".to_string())
        }
    }

    pub fn streaming_asr_finish(&self) -> Result<String, String> {
        let mut text_ptr: *mut i8 = std::ptr::null_mut();

        let result = unsafe { fluidaudio_streaming_asr_finish(self.ptr, &mut text_ptr) };

        if result != 0 {
            return Err("Failed to finish streaming ASR session".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(text)
    }

    pub fn transcribe_file_streaming(&self, path: &str) -> Result<AsrResult, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;

        let mut text_ptr: *mut i8 = std::ptr::null_mut();
        let mut confidence: f32 = 0.0;
        let mut duration: f64 = 0.0;
        let mut processing_time: f64 = 0.0;
        let mut rtfx: f32 = 0.0;

        let result = unsafe {
            fluidaudio_transcribe_file_streaming(
                self.ptr,
                c_path.as_ptr(),
                &mut text_ptr,
                &mut confidence,
                &mut duration,
                &mut processing_time,
                &mut rtfx,
            )
        };

        if result != 0 {
            return Err("Streaming file transcription failed".to_string());
        }

        let text = if text_ptr.is_null() {
            String::new()
        } else {
            let text = unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { fluidaudio_free_string(text_ptr) };
            text
        };

        Ok(AsrResult {
            text,
            confidence,
            duration,
            processing_time,
            rtfx,
        })
    }

    pub fn is_streaming_asr_available(&self) -> bool {
        unsafe { fluidaudio_is_streaming_asr_available(self.ptr) != 0 }
    }

    pub fn initialize_vad(&self, threshold: f32) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_vad(self.ptr, threshold) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize VAD".to_string())
        }
    }

    pub fn is_vad_available(&self) -> bool {
        unsafe { fluidaudio_is_vad_available(self.ptr) != 0 }
    }

    pub fn initialize_diarization(&self, threshold: f64) -> Result<(), String> {
        let result = unsafe { fluidaudio_initialize_diarization(self.ptr, threshold) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to initialize diarization".to_string())
        }
    }

    pub fn diarize_file(&self, path: &str) -> Result<Vec<DiarizationSegment>, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;

        let mut speaker_ids_ptr: *mut *mut i8 = std::ptr::null_mut();
        let mut start_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut end_times_ptr: *mut f32 = std::ptr::null_mut();
        let mut quality_scores_ptr: *mut f32 = std::ptr::null_mut();
        let mut count: u32 = 0;

        let result = unsafe {
            fluidaudio_diarize_file(
                self.ptr,
                c_path.as_ptr(),
                &mut speaker_ids_ptr,
                &mut start_times_ptr,
                &mut end_times_ptr,
                &mut quality_scores_ptr,
                &mut count,
            )
        };

        if result != 0 {
            return Err("Diarization failed".to_string());
        }

        Ok(unsafe {
            collect_diarization_segments(
                speaker_ids_ptr,
                start_times_ptr,
                end_times_ptr,
                quality_scores_ptr,
                count,
            )
        })
    }

    pub fn is_diarization_available(&self) -> bool {
        unsafe { fluidaudio_is_diarization_available(self.ptr) != 0 }
    }

    pub fn system_info(&self) -> SystemInfo {
        let mut platform_ptr: *mut i8 = std::ptr::null_mut();
        let mut chip_ptr: *mut i8 = std::ptr::null_mut();

        unsafe {
            fluidaudio_get_platform(&mut platform_ptr);
            fluidaudio_get_chip_name(&mut chip_ptr);
        }

        let platform = unsafe {
            if platform_ptr.is_null() {
                "unknown".to_string()
            } else {
                let s = CStr::from_ptr(platform_ptr).to_string_lossy().into_owned();
                fluidaudio_free_string(platform_ptr);
                s
            }
        };

        let chip_name = unsafe {
            if chip_ptr.is_null() {
                "unknown".to_string()
            } else {
                let s = CStr::from_ptr(chip_ptr).to_string_lossy().into_owned();
                fluidaudio_free_string(chip_ptr);
                s
            }
        };

        let memory_gb = unsafe { fluidaudio_get_memory_gb() };
        let is_apple_silicon = unsafe { fluidaudio_is_apple_silicon() != 0 };

        SystemInfo {
            platform,
            chip_name,
            memory_gb,
            is_apple_silicon,
        }
    }

    pub fn is_apple_silicon(&self) -> bool {
        unsafe { fluidaudio_is_apple_silicon() != 0 }
    }

    pub fn is_intel_mac(&self) -> bool {
        unsafe { fluidaudio_is_intel_mac() != 0 }
    }

    pub fn vad_process_file(&self, path: &str) -> Result<Vec<VadFrame>, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path")?;

        let mut probs_ptr: *mut f32 = std::ptr::null_mut();
        let mut voice_ptr: *mut u8 = std::ptr::null_mut();
        let mut times_ptr: *mut f64 = std::ptr::null_mut();
        let mut count: u32 = 0;

        let status = unsafe {
            fluidaudio_vad_process_file(
                self.ptr,
                c_path.as_ptr(),
                &mut probs_ptr,
                &mut voice_ptr,
                &mut times_ptr,
                &mut count,
            )
        };

        if status != 0 {
            return Err("VAD process file failed".to_string());
        }

        Ok(unsafe { collect_vad_frames(probs_ptr, voice_ptr, times_ptr, count) })
    }

    pub fn vad_process_samples(&self, samples: &[f32]) -> Result<Vec<VadFrame>, String> {
        let mut probs_ptr: *mut f32 = std::ptr::null_mut();
        let mut voice_ptr: *mut u8 = std::ptr::null_mut();
        let mut times_ptr: *mut f64 = std::ptr::null_mut();
        let mut count: u32 = 0;

        let status = unsafe {
            fluidaudio_vad_process_samples(
                self.ptr,
                samples.as_ptr(),
                samples.len() as u32,
                &mut probs_ptr,
                &mut voice_ptr,
                &mut times_ptr,
                &mut count,
            )
        };

        if status != 0 {
            return Err("VAD process samples failed".to_string());
        }

        Ok(unsafe { collect_vad_frames(probs_ptr, voice_ptr, times_ptr, count) })
    }

    pub fn itn_normalize(&self, text: &str) -> Result<String, String> {
        let c_text = CString::new(text).map_err(|_| "Invalid text (NUL byte)")?;
        let mut out_ptr: *mut i8 = std::ptr::null_mut();
        let status = unsafe { fluidaudio_itn_normalize(self.ptr, c_text.as_ptr(), &mut out_ptr) };
        if status != 0 {
            return Err("ITN normalize failed".to_string());
        }
        Ok(unsafe { take_c_string(out_ptr) })
    }

    pub fn itn_normalize_sentence(&self, text: &str) -> Result<String, String> {
        let c_text = CString::new(text).map_err(|_| "Invalid text (NUL byte)")?;
        let mut out_ptr: *mut i8 = std::ptr::null_mut();
        let status =
            unsafe { fluidaudio_itn_normalize_sentence(self.ptr, c_text.as_ptr(), &mut out_ptr) };
        if status != 0 {
            return Err("ITN normalize_sentence failed".to_string());
        }
        Ok(unsafe { take_c_string(out_ptr) })
    }

    pub fn itn_normalize_sentence_max_span(
        &self,
        text: &str,
        max_span_tokens: u32,
    ) -> Result<String, String> {
        let c_text = CString::new(text).map_err(|_| "Invalid text (NUL byte)")?;
        let mut out_ptr: *mut i8 = std::ptr::null_mut();
        let status = unsafe {
            fluidaudio_itn_normalize_sentence_max_span(
                self.ptr,
                c_text.as_ptr(),
                max_span_tokens,
                &mut out_ptr,
            )
        };
        if status != 0 {
            return Err("ITN normalize_sentence_max_span failed".to_string());
        }
        Ok(unsafe { take_c_string(out_ptr) })
    }

    pub fn itn_is_native_available(&self) -> bool {
        unsafe { fluidaudio_itn_is_native_available(self.ptr) != 0 }
    }

    pub fn cleanup(&self) {
        unsafe { fluidaudio_cleanup(self.ptr) };
    }
}

/// SAFETY: caller must guarantee the four pointers came from a successful
/// `fluidaudio_diarize_*` call with the matching `count`. The result arrays are
/// freed via `fluidaudio_free_diarization_result` before returning — including on
/// the partial-null path, where the free function null-checks each pointer.
unsafe fn collect_diarization_segments(
    speaker_ids_ptr: *mut *mut i8,
    start_times_ptr: *mut f32,
    end_times_ptr: *mut f32,
    quality_scores_ptr: *mut f32,
    count: u32,
) -> Vec<DiarizationSegment> {
    let mut segments = Vec::with_capacity(count as usize);
    if count == 0 {
        return segments;
    }

    if !speaker_ids_ptr.is_null()
        && !start_times_ptr.is_null()
        && !end_times_ptr.is_null()
        && !quality_scores_ptr.is_null()
    {
        for i in 0..count as usize {
            let id_ptr = *speaker_ids_ptr.add(i);
            let speaker_id = if id_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(id_ptr).to_string_lossy().into_owned()
            };
            segments.push(DiarizationSegment {
                speaker_id,
                start_time: *start_times_ptr.add(i),
                end_time: *end_times_ptr.add(i),
                quality_score: *quality_scores_ptr.add(i),
            });
        }
    }

    fluidaudio_free_diarization_result(
        speaker_ids_ptr,
        start_times_ptr,
        end_times_ptr,
        quality_scores_ptr,
        count,
    );
    segments
}

/// SAFETY: caller must guarantee the four pointers came from a successful
/// `fluidaudio_vad_process_*` call with the matching `count`. Pointers are
/// freed via `fluidaudio_free_vad_result` before returning.
unsafe fn collect_vad_frames(
    probs_ptr: *mut f32,
    voice_ptr: *mut u8,
    times_ptr: *mut f64,
    count: u32,
) -> Vec<VadFrame> {
    if count == 0 || probs_ptr.is_null() || voice_ptr.is_null() || times_ptr.is_null() {
        // Even when count==0 the Swift side may pass NULL pointers; free safely.
        fluidaudio_free_vad_result(probs_ptr, voice_ptr, times_ptr, count);
        return Vec::new();
    }

    let probs = std::slice::from_raw_parts(probs_ptr, count as usize);
    let voice = std::slice::from_raw_parts(voice_ptr, count as usize);
    let times = std::slice::from_raw_parts(times_ptr, count as usize);

    let frames: Vec<VadFrame> = probs
        .iter()
        .zip(voice.iter())
        .zip(times.iter())
        .map(|((&probability, &is_voice), &processing_time)| VadFrame {
            probability,
            is_voice_active: is_voice != 0,
            processing_time,
        })
        .collect();

    fluidaudio_free_vad_result(probs_ptr, voice_ptr, times_ptr, count);
    frames
}

/// SAFETY: `ptr` must be either NULL or a C string allocated by the Swift bridge
/// via `strdup`. Freed via `fluidaudio_free_string` before returning.
unsafe fn take_c_string(ptr: *mut i8) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    fluidaudio_free_string(ptr);
    s
}

impl Drop for FluidAudioBridge {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { fluidaudio_bridge_destroy(self.ptr) };
        }
    }
}

// Result types
#[derive(Debug, Clone)]
pub struct AsrResult {
    pub text: String,
    pub confidence: f32,
    pub duration: f64,
    pub processing_time: f64,
    pub rtfx: f32,
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub platform: String,
    pub chip_name: String,
    pub memory_gb: f64,
    pub is_apple_silicon: bool,
}

/// A speaker segment from diarization
#[derive(Debug, Clone)]
pub struct DiarizationSegment {
    /// Speaker identifier (e.g. "SPEAKER_00", "SPEAKER_01")
    pub speaker_id: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
}

impl DiarizationSegment {
    /// Duration of this segment in seconds
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }
}

/// A single per-chunk VAD frame.
///
/// VAD processes audio in 4096-sample chunks (256 ms at 16 kHz). One `VadFrame`
/// is produced per chunk.
#[derive(Debug, Clone, Copy)]
pub struct VadFrame {
    /// Raw model probability that this chunk contains voice (0.0–1.0).
    pub probability: f32,
    /// Whether `probability` crossed the configured threshold (i.e. the chunk
    /// is classified as voice-active).
    pub is_voice_active: bool,
    /// Wall-clock processing time for this chunk in seconds.
    pub processing_time: f64,
}
