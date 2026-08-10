import FluidAudio
import Foundation

// MARK: - Kokoro TTS C FFI
//
// Native Rust binding for FluidAudio's KokoroAneManager. Mirrors the ASR/diarize
// @_cdecl pattern in FluidAudioBridge.swift: recover the bridge from the opaque
// pointer, call the (synchronous) internal method, marshal the result out.

/// Emit a diagnostic to stderr. `synthesize` returns raw WAV bytes that callers
/// typically stream to stdout (see examples/kokoro.rs), so diagnostics must not
/// go to stdout or they corrupt the audio stream.
private func kokoroLog(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

/// Shared body of both init entry points. `computeUnits` is the kebab-case preset
/// name `TtsComputeUnitPreset(cliValue:)` accepts (`default`, `all-ane`,
/// `cpu-and-gpu`, `cpu-only`). NULL or empty keeps the backend's empirical
/// per-stage mapping. An unrecognised value is a caller bug, so it fails rather
/// than silently synthesising on units the caller did not ask for.
private func initializeKokoro(
    _ ptr: UnsafeMutableRawPointer?,
    _ defaultVoice: UnsafePointer<CChar>?,
    _ lang: UnsafePointer<CChar>?,
    _ computeUnits: UnsafePointer<CChar>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let voice = defaultVoice.map { String(cString: $0) } ?? "af_heart"
    let langString = lang.map { String(cString: $0) } ?? ""
    let unitsString = computeUnits.map { String(cString: $0) } ?? ""

    let preset: TtsComputeUnitPreset
    if unitsString.isEmpty {
        preset = .default
    } else if let parsed = TtsComputeUnitPreset(cliValue: unitsString) {
        preset = parsed
    } else {
        kokoroLog(
            "Kokoro init error: unknown compute-units preset '\(unitsString)' "
                + "(expected one of: \(TtsComputeUnitPreset.allCases.map(\.cliValue).joined(separator: ", ")))")
        return -1
    }

    do {
        try bridge.initializeKokoro(defaultVoice: voice, lang: langString, computeUnits: preset)
        return 0
    } catch {
        kokoroLog("Kokoro init error: \(error)")
        return -1
    }
}

/// Initialize on FluidAudio's empirical per-stage compute-unit mapping.
///
/// Kept at three parameters so the existing C symbol's arity is unchanged —
/// anything linking these entry points from an older build keeps working.
@_cdecl("fluidaudio_initialize_kokoro")
public func fluidaudio_initialize_kokoro(
    _ ptr: UnsafeMutableRawPointer?,
    _ defaultVoice: UnsafePointer<CChar>?,
    _ lang: UnsafePointer<CChar>?
) -> Int32 {
    initializeKokoro(ptr, defaultVoice, lang, nil)
}

/// Same, but pins every pipeline stage to an explicit preset. See
/// `initializeKokoro` for the accepted spellings and the failure contract.
@_cdecl("fluidaudio_initialize_kokoro_with_compute_units")
public func fluidaudio_initialize_kokoro_with_compute_units(
    _ ptr: UnsafeMutableRawPointer?,
    _ defaultVoice: UnsafePointer<CChar>?,
    _ lang: UnsafePointer<CChar>?,
    _ computeUnits: UnsafePointer<CChar>?
) -> Int32 {
    initializeKokoro(ptr, defaultVoice, lang, computeUnits)
}

/// Synthesize `text` with `voice` at `speed`; returns a complete WAV byte buffer
/// (24 kHz mono 16-bit PCM (i16), peak-normalized) via `outBytes`/`outLen`. The caller owns the buffer and must
/// free it with `fluidaudio_kokoro_free_bytes`.
@_cdecl("fluidaudio_kokoro_synthesize")
public func fluidaudio_kokoro_synthesize(
    _ ptr: UnsafeMutableRawPointer?,
    _ text: UnsafePointer<CChar>?,
    _ voice: UnsafePointer<CChar>?,
    _ speed: Float,
    _ outBytes: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outLen: UnsafeMutablePointer<UInt>?
) -> Int32 {
    // Both output pointers are required: without them the caller can neither
    // receive the buffer nor its length, and allocating anyway would leak.
    guard let ptr = ptr, let text = text, let outBytes = outBytes, let outLen = outLen else {
        return -1
    }
    outBytes.pointee = nil
    outLen.pointee = 0
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let textString = String(cString: text)
    let voiceString = voice.map { String(cString: $0) } ?? "af_heart"
    do {
        let data = try bridge.synthesizeKokoro(text: textString, voice: voiceString, speed: speed)
        let count = data.count
        let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        data.copyBytes(to: buf, count: count)
        outBytes.pointee = buf
        outLen.pointee = UInt(count)
        return 0
    } catch {
        kokoroLog("Kokoro synthesize error: \(error)")
        return -1
    }
}

/// Install (or clear) English pronunciation overrides on an initialized Kokoro
/// engine. `words` and `phonemes` are parallel arrays of `count` NUL-terminated
/// UTF-8 strings (word → Misaki-style IPA); Swift copies them, so the caller may
/// free them as soon as this returns. `count == 0` clears the table and ignores
/// both array pointers. Fails when Kokoro has not been initialized — the table
/// lives on the manager, so a silent no-op would leave the caller believing its
/// pronunciations were installed.
@_cdecl("fluidaudio_kokoro_set_english_lexicon")
public func fluidaudio_kokoro_set_english_lexicon(
    _ ptr: UnsafeMutableRawPointer?,
    _ words: UnsafePointer<UnsafePointer<CChar>?>?,
    _ phonemes: UnsafePointer<UnsafePointer<CChar>?>?,
    _ count: UInt
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    var entries: [String: String] = [:]
    if count > 0 {
        guard let words = words, let phonemes = phonemes else {
            kokoroLog("Kokoro lexicon error: \(count) entries requested with a NULL array")
            return -1
        }
        entries.reserveCapacity(Int(count))
        for index in 0..<Int(count) {
            guard let word = words[index], let ipa = phonemes[index] else {
                kokoroLog("Kokoro lexicon error: NULL string at index \(index)")
                return -1
            }
            entries[String(cString: word)] = String(cString: ipa)
        }
    }

    do {
        try bridge.setKokoroEnglishLexicon(entries)
        return 0
    } catch {
        kokoroLog("Kokoro lexicon error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_kokoro_free_bytes")
public func fluidaudio_kokoro_free_bytes(_ p: UnsafeMutablePointer<UInt8>?) {
    p?.deallocate()
}

@_cdecl("fluidaudio_is_kokoro_available")
public func fluidaudio_is_kokoro_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isKokoroAvailable() ? 1 : 0
}
