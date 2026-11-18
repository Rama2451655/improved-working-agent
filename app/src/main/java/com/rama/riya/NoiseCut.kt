package com.rama.riya

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Removes steady background noise (fan, AC, hum, hiss, distant traffic) from the voice.
 * Method: 32 ms frames (50% overlap), the noise of each frequency is tracked as the quietest level of the last 1.5 s,
 * each frequency is turned down by how much of it is noise (never more than -14 dB, so the voice is not damaged),
 * the turn-down is smoothed in time and across frequencies to avoid "bubbling" artefacts.
 * Adds 16 ms delay. Pure Kotlin (no Android classes).
 */
class NoiseCut(private val maxCutDb: Double = 14.0) {
    private val n = 512
    private val hop = 256
    private val bins = n / 2 + 1
    private val win = DoubleArray(n) { sqrt(0.5 - 0.5 * cos(2.0 * PI * it / n)) }      // sqrt-Hann: analysis AND synthesis
    private val cosT = DoubleArray(n / 2) { cos(2.0 * PI * it / n) }
    private val sinT = DoubleArray(n / 2) { -sin(2.0 * PI * it / n) }
    private val re = DoubleArray(n)
    private val im = DoubleArray(n)

    private val inRing = DoubleArray(n)           // last 512 input samples
    private var inCount = 0
    private val outAcc = DoubleArray(n)           // overlap-add accumulator
    private val outQ = java.util.ArrayDeque<Double>()

    private val smoothP = DoubleArray(bins)
    private val subMin = Array(6) { DoubleArray(bins) { Double.MAX_VALUE } }
    private var subIdx = 0
    private var subCount = 0
    private val noise = DoubleArray(bins)
    private var frames = 0
    private val gPrev = DoubleArray(bins) { 1.0 }
    private val gNow = DoubleArray(bins)
    private val gMin = Math.pow(10.0, -maxCutDb / 20.0)

    init { reset() }

    fun reset() {
        java.util.Arrays.fill(inRing, 0.0); inCount = 0; java.util.Arrays.fill(outAcc, 0.0)
        outQ.clear(); repeat(hop) { outQ.add(0.0) }
        java.util.Arrays.fill(smoothP, 0.0); java.util.Arrays.fill(noise, 0.0); frames = 0
        for (s in subMin) java.util.Arrays.fill(s, Double.MAX_VALUE)
        subIdx = 0; subCount = 0; java.util.Arrays.fill(gPrev, 1.0)
    }

    private fun fft(inverse: Boolean) {
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
                    val wr = cosT[k]; val wi = if (inverse) -sinT[k] else sinT[k]
                    val a = i + m; val b = i + m + len / 2
                    val vr = re[b] * wr - im[b] * wi
                    val vi = re[b] * wi + im[b] * wr
                    re[b] = re[a] - vr; im[b] = im[a] - vi
                    re[a] += vr; im[a] += vi
                    k += step
                }
                i += len
            }
            len = len shl 1
        }
    }

    private fun frame() {
        syncRing()
        for (i in 0 until n) { re[i] = inRing[i] * win[i]; im[i] = 0.0 }
        fft(false)
        // power, smoothed in time
        for (k in 0 until bins) {
            val p = re[k] * re[k] + im[k] * im[k]
            smoothP[k] = if (frames == 0) p else smoothP[k] * 0.7 + p * 0.3
        }
        // noise = quietest smoothed level of the last ~1.5 s (6 blocks of 16 frames), times a bias factor
        for (k in 0 until bins) if (smoothP[k] < subMin[subIdx][k]) subMin[subIdx][k] = smoothP[k]
        if (++subCount >= 16) {
            subCount = 0
            subIdx = (subIdx + 1) % subMin.size
            java.util.Arrays.fill(subMin[subIdx], Double.MAX_VALUE)
        }
        for (k in 0 until bins) {
            var m = Double.MAX_VALUE
            for (s in subMin) if (s[k] < m) m = s[k]
            noise[k] = if (m == Double.MAX_VALUE) smoothP[k] else m * 2.0
        }
        frames++
        // gain per frequency: Wiener-like with over-subtraction, floored
        for (k in 0 until bins) {
            val p = re[k] * re[k] + im[k] * im[k]
            val snrPost = p / (noise[k] + 1e-9)
            var g = 1.0 - 1.4 / max(snrPost, 1e-6)
            if (g < gMin) g = gMin
            if (g > 1.0) g = 1.0
            gNow[k] = g
        }
        // smooth over frequency (3 bins) and time
        for (k in 0 until bins) {
            val a = gNow[max(0, k - 1)]; val b = gNow[k]; val c = gNow[min(bins - 1, k + 1)]
            val gf = 0.25 * a + 0.5 * b + 0.25 * c
            val g = if (gf < gPrev[k]) gf * 0.6 + gPrev[k] * 0.4 else gf * 0.8 + gPrev[k] * 0.2     // fall fast, rise a bit slower
            gPrev[k] = g
            re[k] *= g; im[k] *= g
            if (k in 1 until n / 2) { re[n - k] = re[k]; im[n - k] = -im[k] }
        }
        fft(true)
        for (i in 0 until n) outAcc[i] += re[i] / n * win[i]
        for (i in 0 until hop) outQ.add(outAcc[i])
        System.arraycopy(outAcc, hop, outAcc, 0, n - hop)
        java.util.Arrays.fill(outAcc, n - hop, n, 0.0)
    }

    private var wr = 0
    private val ring = DoubleArray(n)

    /** the last 512 samples, oldest first */
    private fun syncRing() {
        for (i in 0 until n) inRing[i] = ring[(wr + i) % n]
    }

    /** Cleans [len] samples in place (the sound comes out 16 ms later than it went in). */
    fun process(buf: ShortArray, len: Int) {
        for (i in 0 until len) {
            ring[wr] = buf[i].toDouble()
            wr = (wr + 1) % n
            if (++inCount == hop) { inCount = 0; frame() }
            val o = outQ.pollFirst() ?: 0.0
            buf[i] = (if (o > 32767.0) 32767.0 else if (o < -32768.0) -32768.0 else o).toInt().toShort()
        }
    }
}
