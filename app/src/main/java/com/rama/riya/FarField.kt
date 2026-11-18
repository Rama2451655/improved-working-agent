package com.rama.riya

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Hearing from far away, and only people.
 *  1. High-pass filter (about 150 Hz): the rumble of traffic, a fan, an AC or the table is removed.
 *  2. Voice detector: a sound counts as a human voice only when (a) its 300-3400 Hz energy is clearly above the room's
 *     own steady noise (the room noise is learned all the time) and (b) it has a pitch (the regular vibration of vocal
 *     cords, which fans, AC and engines do not have).
 *  3. Automatic gain: a quiet far voice is lifted to a normal level (up to 12x); sound that is not a voice is turned down.
 * Pure Kotlin (no Android classes) so it can be tested on a computer.
 */
class FarField(private val rate: Int = 16000, removeNoise: Boolean = true) {

    private val cut: NoiseCut? = if (removeNoise) NoiseCut() else null

    /** the last chunk of audio contained a human-voice-like sound (includes a short hang-over) */
    @Volatile var voice = false
        private set

    // ---- high-pass filter: 2nd order Butterworth, 150 Hz ----
    private val b0: Double; private val b1: Double; private val b2: Double
    private val a1: Double; private val a2: Double
    private var z1 = 0.0
    private var z2 = 0.0

    // ---- spectrum analysis ----
    private val n = 512
    private val win = DoubleArray(n) { 0.5 - 0.5 * cos(2.0 * PI * it / (n - 1)) }
    private val cosT = DoubleArray(n / 2) { cos(2.0 * PI * it / n) }
    private val sinT = DoubleArray(n / 2) { -sin(2.0 * PI * it / n) }
    private val re = DoubleArray(n)
    private val im = DoubleArray(n)
    private val noise = DoubleArray(n / 2 + 1)
    private var noiseFrames = 0
    private val frame = DoubleArray(n)
    private val pre = DoubleArray(n)
    private var filled = 0

    // ---- voice decision and gain ----
    private var hold = 0
    private var quiet = 0             // chunks since the last voice
    private var voiceRun = 0          // frames in a row judged "voice": a real voice pauses, a steady hum does not
    private var level = 0.0          // typical loudness of the voice (after the filter)
    private var gain = 1.0

    init {
        val w0 = 2.0 * PI * 150.0 / rate
        val alpha = sin(w0) / (2.0 * 0.7071)
        val c = cos(w0)
        val a0 = 1.0 + alpha
        b0 = ((1.0 + c) / 2.0) / a0
        b1 = (-(1.0 + c)) / a0
        b2 = ((1.0 + c) / 2.0) / a0
        a1 = (-2.0 * c) / a0
        a2 = (1.0 - alpha) / a0
    }

    fun reset() {
        cut?.reset(); z1 = 0.0; z2 = 0.0; filled = 0; noiseFrames = 0; hold = 0; voiceRun = 0; level = 0.0; gain = 1.0; voice = false
        java.util.Arrays.fill(noise, 0.0)
    }

    private fun fft() {
        // bit reversal
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { val tr = re[i]; re[i] = re[j]; re[j] = tr; val ti = im[i]; im[i] = im[j]; im[j] = ti }
        }
        var len = 2
        while (len <= n) {
            val step = n / len
            var i = 0
            while (i < n) {
                var k = 0
                for (m in 0 until len / 2) {
                    val wr = cosT[k]; val wi = sinT[k]
                    val ur = re[i + m]; val ui = im[i + m]
                    val vr = re[i + m + len / 2] * wr - im[i + m + len / 2] * wi
                    val vi = re[i + m + len / 2] * wi + im[i + m + len / 2] * wr
                    re[i + m] = ur + vr; im[i + m] = ui + vi
                    re[i + m + len / 2] = ur - vr; im[i + m + len / 2] = ui - vi
                    k += step
                }
                i += len
            }
            len = len shl 1
        }
    }

    /** Is this 32 ms frame a human voice? Also keeps learning the steady room noise. */
    private fun frameIsVoice(x: DoubleArray): Boolean {
        var e = 0.0
        for (i in 0 until n) { re[i] = x[i] * win[i]; im[i] = 0.0; e += x[i] * x[i] }
        val rms = sqrt(e / n) / 32768.0
        fft()
        val binHz = rate.toDouble() / n
        val lo = (300.0 / binHz).toInt()
        val hi = (3400.0 / binHz).toInt()
        var sig = 0.0
        var noi = 0.0
        var tot = 1e-9
        for (k in 0..n / 2) {
            val p = re[k] * re[k] + im[k] * im[k]
            tot += p
            if (noiseFrames < 12) noise[k] = if (noiseFrames == 0) p else noise[k] * 0.8 + p * 0.2
            if (k in lo..hi) { sig += p; noi += noise[k] }
        }
        val snr = sig / (noi + 1e-9)
        val share = sig / tot            // a voice has most of its sound in 300-3400 Hz; a rumble does not
        var voiced = false
        if (rms > 0.0002 && noiseFrames >= 12) {          // a very quiet far voice (-74 dB full scale) still counts
            // pitch: the signal repeats itself every 2.5 - 12.5 ms (80 - 400 Hz) in a voice
            // (looked at after a pre-emphasis, so a low rumble or hum cannot pass for a voice: only the voice's upper harmonics count)
            pre[0] = x[0]
            for (i in 1 until n) pre[i] = x[i] - 0.0 * x[i - 1]
            var r0 = 1e-9
            for (i in 0 until n) r0 += pre[i] * pre[i]
            var best = 0.0
            for (lag in 40..200) {
                var r = 0.0
                for (i in 0 until n - lag) r += pre[i] * pre[i + lag]
                val v = r / r0
                if (v > best) best = v
            }
            voiced = share > 0.25 && ((snr > 2.2 && best > 0.28) || snr > 14.0)
        }
        // a "voice" that never pauses for 4 seconds is a steady sound (a hum, an alarm): learn it as room noise
        if (voiced) { if (++voiceRun > 125) voiced = false } else voiceRun = 0
        // learn the room noise (its average power) from frames that are not a voice; frozen while someone talks
        val rate1 = if (voiced) 0.0005 else 0.03
        if (noiseFrames >= 12) for (k in 0..n / 2) {
            val p = re[k] * re[k] + im[k] * im[k]
            noise[k] += (p - noise[k]) * rate1
        }
        if (noiseFrames < 12) noiseFrames++
        return voiced
    }

    /**
     * Cleans one chunk in place: rumble removed, quiet voice lifted, non-voice sound turned down.
     * [n] samples of 16-bit audio. After the call [voice] tells whether a human voice was in it.
     */
    fun process(buf: ShortArray, len: Int) {
        // 1. high-pass
        for (i in 0 until len) {
            val x = buf[i].toDouble()
            val y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            buf[i] = y.coerceIn(-32768.0, 32767.0).toInt().toShort()
        }
        // 2. voice detection on the filtered sound (frames of 512 samples, may span two chunks)
        var anyVoice = false
        var sumSq = 0.0
        for (i in 0 until len) {
            val v = buf[i].toDouble()
            sumSq += v * v
            frame[filled++] = v
            if (filled == n) {
                if (frameIsVoice(frame)) anyVoice = true
                filled = 0
            }
        }
        cut?.process(buf, len)                  // steady noise turned down in every frequency (the voice detector above looked at the sound before this)
        sumSq = 0.0
        for (i in 0 until len) { val v = buf[i].toDouble(); sumSq += v * v }
        val chunkRms = sqrt(sumSq / len.coerceAtLeast(1)) / 32768.0
        if (anyVoice) { hold = 8; quiet = 0 } else { if (hold > 0) hold--; quiet++ }
        voice = hold > 0
        // 3. gain: follow the voice level; the voice is lifted to about 6% of full scale, other sound is turned down
        if (anyVoice) level = if (level == 0.0) chunkRms else level + (chunkRms - level) * (if (chunkRms > level) 0.4 else 0.05)
        val target = if (level > 0.0) (0.06 / level).coerceIn(1.0, 30.0) else 1.0
        gain += (target - gain) * (if (target < gain) 0.3 else 0.08)
        val g = if (quiet > 20) gain * 0.4 else gain     // sound that is not a voice is turned down, but only after 1.5 s without one
        for (i in 0 until len) {
            val y = buf[i] * g
            buf[i] = (if (y > 32000.0) 32000.0 else if (y < -32000.0) -32000.0 else y).toInt().toShort()
        }
    }
}
