//! Example: synthesize TTS via FluidAudio Kokoro.
//! Usage: cargo run --example kokoro --features tts -- "Hello world" af_heart en-us > out.wav
//!        cargo run --example kokoro --features tts -- "你好" zf_001 zh > out.wav
//!        cargo run --example kokoro --features tts -- "Hi" af_heart en-us cpu-and-gpu > out.wav
//! `lang` selects the KokoroAne variant (`zh` → Mandarin, else English).
//! The optional 4th arg picks CoreML compute units (`default`, `all-ane`,
//! `cpu-and-gpu`, `cpu-only`). Hosts without a usable Neural Engine — a
//! virtualised macOS guest, e.g. a GitHub-hosted `macos-14` runner — need
//! `cpu-and-gpu`; the default mapping pins four stages to the ANE and fails
//! there with "Failed to prepare the model for predictions".
//! `af_heart` (English) and `zf_001` (Mandarin) are the built-in default voices.
//! Note: English currently hosts only `af_heart`; Mandarin hosts ~100 voices (`zf_*`, `zm_*`, …)
//! that download on demand from HuggingFace on first use. An id absent from the hosted bundle
//! fails to load — or pre-stage it as `<voice>.bin` in the model cache.
use fluidaudio_rs::{FluidAudio, KokoroComputeUnits};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let text = args.get(1).map(String::as_str).unwrap_or("Hello world");
    let voice = args.get(2).map(String::as_str).unwrap_or("af_heart");
    let lang = args.get(3).map(String::as_str).unwrap_or("en-us");
    let compute_units: KokoroComputeUnits =
        args.get(4).map_or(Ok(Default::default()), |s| s.parse())?;

    let audio = FluidAudio::new()?;
    eprintln!(
        "Initializing Kokoro on {} compute units (downloads model on first run)...",
        compute_units.as_str()
    );
    audio.init_kokoro_with_compute_units(voice, lang, compute_units)?;
    eprintln!("Kokoro available: {}", audio.is_kokoro_available());

    let wav = audio.synthesize_kokoro(text, voice, 1.0)?;
    eprintln!("WAV bytes: {}", wav.len());
    std::io::stdout().write_all(&wav)?;
    Ok(())
}
