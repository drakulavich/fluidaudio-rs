use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Tell Cargo to rerun if Swift files change
    println!("cargo:rerun-if-changed=swift/");
    println!("cargo:rerun-if-changed=Package.swift");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Build the Swift package first to get FluidAudio dependency
    println!("cargo:warning=Building Swift package...");

    let swift_build_dir = out_dir.join("swift-build");
    std::fs::create_dir_all(&swift_build_dir).expect("Failed to create swift-build directory");

    // Build Swift package in release mode
    let status = Command::new("swift")
        .args(&[
            "build",
            "-c",
            "release",
            "--build-path",
            swift_build_dir.to_str().unwrap(),
        ])
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to run swift build");

    if !status.success() {
        panic!("Swift package build failed");
    }

    // Find the built library
    let lib_path = swift_build_dir.join("release");

    // Link the Swift library
    println!("cargo:rustc-link-search=native={}", lib_path.display());

    // SwiftPM 6.0.3 (Xcode 16.2) leaves FluidAudio's NeMo staticlib (#867) out of the bridge archive.
    if !archive_defines(&lib_path.join("libFluidAudioBridge.a"), "_nemo_normalize") {
        let nemo_dir = nemo_archive_dir(&swift_build_dir);
        println!(
            "cargo:warning=SwiftPM left the NeMo staticlib out of libFluidAudioBridge.a; linking {}",
            nemo_dir.join("libtext_processing_rs.a").display()
        );
        println!("cargo:rustc-link-search=native={}", nemo_dir.display());
        // Build-script -l flags follow #[link] libraries, so this lands after -lFluidAudioBridge.
        println!("cargo:rustc-link-lib=static:-bundle=text_processing_rs");
    }

    // Link Apple frameworks
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Link Swift runtime
    println!("cargo:rustc-link-lib=dylib=swiftCore");

    // Link C++ standard library (needed for FastClusterWrapper.cpp in FluidAudio)
    println!("cargo:rustc-link-lib=c++");
}

/// Only nm's stdout counts: it exits non-zero on members whose embedded bitcode comes from a
/// newer LLVM than Xcode's (the staticlib's std, built by rustc 1.97) while still listing the rest.
fn archive_defines(archive: &Path, symbol: &str) -> bool {
    let output = Command::new("nm")
        .args(["-gU", archive.to_str().unwrap()])
        .output()
        .expect("Failed to run nm (part of the Xcode command line tools)");
    let definition = format!(" T {symbol}");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|line| line.ends_with(&definition))
}

/// The macOS slice of the unpacked NemoTextProcessing.xcframework under SwiftPM's artifacts.
fn nemo_archive_dir(swift_build_dir: &Path) -> PathBuf {
    let artifacts = swift_build_dir.join("artifacts");
    let framework =
        find_dir_named(&artifacts, "NemoTextProcessing.xcframework").unwrap_or_else(|| {
            panic!(
                "libFluidAudioBridge.a does not define _nemo_normalize and no \
             NemoTextProcessing.xcframework exists under {}: FluidAudio's NeMo staticlib must be \
             linked one way or the other (swift package resolve should have unpacked it)",
                artifacts.display()
            )
        });
    std::fs::read_dir(&framework)
        .expect("read the xcframework directory")
        .flatten()
        .map(|entry| entry.path())
        .find(|slice| {
            slice
                .file_name()
                .and_then(|name| name.to_str())
                .map_or(false, |name| name.starts_with("macos-"))
                && slice.join("libtext_processing_rs.a").is_file()
        })
        .unwrap_or_else(|| {
            panic!(
                "{} has no macos-* slice containing libtext_processing_rs.a",
                framework.display()
            )
        })
}

fn find_dir_named(root: &Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(path);
        }
        if let Some(found) = find_dir_named(&path, name) {
            return Some(found);
        }
    }
    None
}
