//! Example: synthesize TTS via FluidAudio Kokoro.
//! Usage: cargo run --example kokoro --features tts -- "Hello world" af_heart en-us > out.wav
//!        cargo run --example kokoro --features tts -- "你好" zf_001 zh > out.wav
//! `lang` selects the KokoroAne variant (`zh` → Mandarin, else English).
//! `af_heart` (English) and `zf_001` (Mandarin) are the built-in default voices.
//! Note: English currently hosts only `af_heart`; Mandarin hosts ~100 voices (`zf_*`, `zm_*`, …)
//! that download on demand from HuggingFace on first use. An id absent from the hosted bundle
//! fails to load — or pre-stage it as `<voice>.bin` in the model cache.
use fluidaudio_rs::FluidAudio;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let text = args.get(1).map(String::as_str).unwrap_or("Hello world");
    let voice = args.get(2).map(String::as_str).unwrap_or("af_heart");
    let lang = args.get(3).map(String::as_str).unwrap_or("en-us");

    let audio = FluidAudio::new()?;
    eprintln!("Initializing Kokoro (downloads model on first run)...");
    audio.init_kokoro(voice, lang)?;
    eprintln!("Kokoro available: {}", audio.is_kokoro_available());

    let wav = audio.synthesize_kokoro(text, voice, 1.0)?;
    eprintln!("WAV bytes: {}", wav.len());
    std::io::stdout().write_all(&wav)?;
    Ok(())
}
