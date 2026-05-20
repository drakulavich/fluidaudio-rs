import Foundation

// MARK: - Model-path diarization C FFI
//
// Diarize from a pre-staged Sortformer .mlpackage (no HuggingFace download), for
// callers that pin/verify the model themselves. Output marshalling mirrors
// fluidaudio_diarize_file; the caller frees the arrays with
// fluidaudio_free_diarization_result.

@_cdecl("fluidaudio_diarize_file_with_models")
public func fluidaudio_diarize_file_with_models(
    _ ptr: UnsafeMutableRawPointer?,
    _ audioPath: UnsafePointer<CChar>?,
    _ modelPath: UnsafePointer<CChar>?,
    _ outSpeakerIds: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?>?,
    _ outStartTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outEndTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outQualityScores: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let audioPath = audioPath, let modelPath = modelPath else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let audioString = String(cString: audioPath)
    let modelString = String(cString: modelPath)

    do {
        let segments = try bridge.diarizeFileWithModels(audioPath: audioString, modelPath: modelString)
        let count = segments.count

        outCount?.pointee = UInt32(count)

        if count == 0 {
            outSpeakerIds?.pointee = nil
            outStartTimes?.pointee = nil
            outEndTimes?.pointee = nil
            outQualityScores?.pointee = nil
        } else {
            let ids = UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>.allocate(capacity: count)
            let starts = UnsafeMutablePointer<Float>.allocate(capacity: count)
            let ends = UnsafeMutablePointer<Float>.allocate(capacity: count)
            let scores = UnsafeMutablePointer<Float>.allocate(capacity: count)

            for (i, seg) in segments.enumerated() {
                ids[i] = strdup(seg.speakerId)
                starts[i] = seg.startTime
                ends[i] = seg.endTime
                scores[i] = seg.qualityScore
            }

            outSpeakerIds?.pointee = ids
            outStartTimes?.pointee = starts
            outEndTimes?.pointee = ends
            outQualityScores?.pointee = scores
        }

        return 0
    } catch {
        print("Diarize (model path) error: \(error)")
        return -1
    }
}
