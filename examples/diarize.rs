//! Example: Diarize an audio file (identify speakers)
//!
//! Usage: cargo run --example diarize -- path/to/audio.wav [threshold] [model.mlpackage] [compute-units]
//!
//! With a 3rd arg (a pre-staged Sortformer `.mlpackage`), diarization runs offline via
//! `diarize_file_with_models` with no download; `threshold` is ignored in that mode.
//! A 4th arg (`all`, `cpu-and-ane`, `cpu-and-gpu`, `cpu-only`) switches to the
//! controlled API, which also prints per-chunk progress.

use fluidaudio_rs::{DiarizeComputeUnits, DiarizeOutcome, DiarizeProgress, FluidAudio};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let audio_path = args
        .get(1)
        .ok_or("Usage: diarize <audio_file> [threshold]")?;
    let threshold: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.6);

    println!("FluidAudio Diarization Example");
    println!("==============================\n");

    let audio = FluidAudio::new()?;

    let info = audio.system_info();
    println!("System: {} ({})", info.chip_name, info.platform);
    println!("Apple Silicon: {}\n", audio.is_apple_silicon());

    // If a model path (.mlpackage) is given as the 3rd arg, use the no-download
    // model-path API; otherwise fall back to the auto-downloading path.
    println!("Diarizing: {}", audio_path);
    let segments = if let (Some(model_path), Some(units)) = (args.get(3), args.get(4)) {
        let units: DiarizeComputeUnits = units.parse()?;
        println!(
            "Using pre-staged Sortformer model on {}: {model_path}",
            units.as_str()
        );
        let mut on_progress = |p: DiarizeProgress| {
            let pct = p.processed_samples as f64 / p.total_samples.max(1) as f64 * 100.0;
            eprintln!("  chunk {} — {pct:.1}%", p.chunks);
        };
        match audio.diarize_file_with_models_controlled(
            audio_path,
            model_path,
            units,
            None,
            Some(&mut on_progress),
        )? {
            DiarizeOutcome::Completed(segments) => segments,
            DiarizeOutcome::Cancelled => return Err("diarization was cancelled".into()),
        }
    } else if let Some(model_path) = args.get(3) {
        println!("Using pre-staged Sortformer model (no download): {model_path}");
        audio.diarize_file_with_models(audio_path, model_path)?
    } else {
        println!(
            "Initializing diarization (threshold={:.2}, downloads on first run)...",
            threshold
        );
        audio.init_diarization(threshold)?;
        audio.diarize_file(audio_path)?
    };

    println!("\n--- Results ({} segments) ---\n", segments.len());
    for seg in &segments {
        println!(
            "[{:.2}s - {:.2}s] {} (quality: {:.2})",
            seg.start_time, seg.end_time, seg.speaker_id, seg.quality_score
        );
    }

    Ok(())
}
