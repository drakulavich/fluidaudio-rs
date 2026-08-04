import Foundation

// MARK: - Kokoro TTS C FFI
//
// Native Rust binding for FluidAudio's KokoroAneManager. Mirrors the ASR/diarize
// @_cdecl pattern in FluidAudioBridge.swift: recover the bridge from the opaque
// pointer, call the (synchronous) internal method, marshal the result out.

@_cdecl("fluidaudio_initialize_kokoro")
public func fluidaudio_initialize_kokoro(
    _ ptr: UnsafeMutableRawPointer?,
    _ defaultVoice: UnsafePointer<CChar>?,
    _ lang: UnsafePointer<CChar>?
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let voice = defaultVoice.map { String(cString: $0) } ?? "af_heart"
    let langString = lang.map { String(cString: $0) } ?? ""
    do {
        try bridge.initializeKokoro(defaultVoice: voice, lang: langString)
        return 0
    } catch {
        print("Kokoro init error: \(error)")
        return -1
    }
}

/// Same as `fluidaudio_initialize_kokoro`, but overrides the CoreML compute units for every
/// pipeline stage: 1 = cpuAndGpu (skips the Neural Engine entirely), 2 = allAne, 3 = cpuOnly,
/// anything else = FluidAudio's per-stage defaults. The escape hatch matters on hosts with no
/// ANE — a virtualised macOS runner cannot prepare the ANE-pinned vocoder stage at all.
@_cdecl("fluidaudio_initialize_kokoro_with_compute_units")
public func fluidaudio_initialize_kokoro_with_compute_units(
    _ ptr: UnsafeMutableRawPointer?,
    _ defaultVoice: UnsafePointer<CChar>?,
    _ lang: UnsafePointer<CChar>?,
    _ computeUnits: Int32
) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let voice = defaultVoice.map { String(cString: $0) } ?? "af_heart"
    let langString = lang.map { String(cString: $0) } ?? ""
    do {
        try bridge.initializeKokoro(
            defaultVoice: voice, lang: langString,
            computeUnits: FluidAudioBridgeInternal.kokoroComputeUnits(for: computeUnits))
        return 0
    } catch {
        print("Kokoro init error: \(error)")
        return -1
    }
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
    guard let ptr = ptr, let text = text else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let textString = String(cString: text)
    let voiceString = voice.map { String(cString: $0) } ?? "af_heart"
    do {
        let data = try bridge.synthesizeKokoro(text: textString, voice: voiceString, speed: speed)
        let count = data.count
        let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        data.copyBytes(to: buf, count: count)
        outBytes?.pointee = buf
        outLen?.pointee = UInt(count)
        return 0
    } catch {
        print("Kokoro synthesize error: \(error)")
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
