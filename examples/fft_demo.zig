//! Demonstration of FFT (Fast Fourier Transform) operations
//!
//! Run with: `zig build run-fft`
//!
//! This example shows:
//! 1. 1D FFT and inverse FFT
//! 2. Magnitude and power spectra
//! 3. Signal analysis with FFT

const std = @import("std");
const zpu = @import("zpu");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    
    try zpu.init(alloc, .{});
    defer zpu.deinit();
    
    std.debug.print("=== FFT Operations Demo ===\n\n", .{});
    
    // 1. Simple FFT test
    std.debug.print("1. Basic FFT\n", .{});
    std.debug.print("------------\n", .{});
    
    const signal1 = try zpu.Tensor.fromArray(f64, &.{ 1.0, 0.0, 0.0, 0.0 }, try zpu.Shape.init(&.{4}));
    defer signal1.deinit();
    
    std.debug.print("Input: [1, 0, 0, 0]\n", .{});
    
    const fft_result = try zpu.fft.fft1d(signal1);
    defer fft_result.deinit();
    
    std.debug.print("FFT result (complex):\n", .{});
    for (0..4) |i| {
        const re = fft_result.readF64(2 * i);
        const im = fft_result.readF64(2 * i + 1);
        std.debug.print("  X[{d}] = {d:.3} + {d:.3}i\n", .{ i, re, im });
    }
    
    // 2. FFT of a sine wave
    std.debug.print("\n2. FFT of Sine Wave\n", .{});
    std.debug.print("-------------------\n", .{});
    
    // Generate a sine wave: sin(2π * 2 * t) for t in [0, 1)
    const n_samples = 16;
    var sine_data: [16]f64 = undefined;
    for (0..n_samples) |i| {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_samples));
        sine_data[i] = @sin(2.0 * std.math.pi * 2.0 * t);
    }
    
    const sine_wave = try zpu.Tensor.fromArray(f64, &sine_data, try zpu.Shape.init(&.{16}));
    defer sine_wave.deinit();
    
    std.debug.print("Generated sine wave with frequency 2 (16 samples)\n", .{});
    std.debug.print("First 8 samples: [", .{});
    for (0..8) |i| {
        std.debug.print("{d:.2}", .{sine_wave.readF64(i)});
        if (i < 7) std.debug.print(", ", .{});
    }
    std.debug.print("]\n", .{});
    
    const sine_fft = try zpu.fft.fft1d(sine_wave);
    defer sine_fft.deinit();
    
    // Find the peak frequency
    var max_mag: f64 = 0;
    var peak_idx: usize = 0;
    for (0..n_samples) |i| {
        const re = sine_fft.readF64(2 * i);
        const im = sine_fft.readF64(2 * i + 1);
        const mag = @sqrt(re * re + im * im);
        if (mag > max_mag) {
            max_mag = mag;
            peak_idx = i;
        }
    }
    
    std.debug.print("Peak frequency index: {d} (expected 2 or 14)\n", .{peak_idx});
    std.debug.print("Peak magnitude: {d:.3}\n", .{max_mag});
    
    // 3. Magnitude spectrum
    std.debug.print("\n3. Magnitude Spectrum\n", .{});
    std.debug.print("---------------------\n", .{});
    
    const magnitude = try zpu.fft.fftMag(sine_wave);
    defer magnitude.deinit();
    
    std.debug.print("Magnitude spectrum:\n", .{});
    for (0..n_samples) |i| {
        std.debug.print("  |X[{d:2}]| = {d:.3}\n", .{ i, magnitude.readF64(i) });
    }
    
    // 4. Power spectrum
    std.debug.print("\n4. Power Spectrum\n", .{});
    std.debug.print("-----------------\n", .{});
    
    const power = try zpu.fft.fftPower(sine_wave);
    defer power.deinit();
    
    std.debug.print("Power spectrum (first 8 bins):\n", .{});
    for (0..8) |i| {
        std.debug.print("  |X[{d}]|^2 = {d:.3}\n", .{ i, power.readF64(i) });
    }
    
    // 5. Inverse FFT test
    std.debug.print("\n5. Inverse FFT\n", .{});
    std.debug.print("--------------\n", .{});
    
    const recovered = try zpu.fft.ifft1d(sine_fft);
    defer recovered.deinit();
    
    std.debug.print("Original vs Recovered (first 8 samples):\n", .{});
    var max_error: f64 = 0;
    for (0..8) |i| {
        const orig = sine_wave.readF64(i);
        const rec_re = recovered.readF64(2 * i);
        const error = @abs(orig - rec_re);
        if (error > max_error) max_error = error;
        
        std.debug.print("  [{d}] orig={d:.3}, recovered={d:.3}, error={d:.6}\n", .{ i, orig, rec_re, error });
    }
    std.debug.print("Max reconstruction error: {d:.9}\n", .{max_error});
    
    // 6. Practical application: Signal filtering
    std.debug.print("\n6. Application: Low-pass Filter\n", .{});
    std.debug.print("-------------------------------\n", .{});
    
    // Create a signal with two frequencies
    var mixed_data: [16]f64 = undefined;
    for (0..n_samples) |i| {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_samples));
        // Mix of frequency 1 and frequency 4
        mixed_data[i] = @sin(2.0 * std.math.pi * 1.0 * t) + 0.5 * @sin(2.0 * std.math.pi * 4.0 * t);
    }
    
    const mixed_signal = try zpu.Tensor.fromArray(f64, &mixed_data, try zpu.Shape.init(&.{16}));
    defer mixed_signal.deinit();
    
    std.debug.print("Mixed signal: freq 1 + 0.5 * freq 4\n", .{});
    
    const mixed_fft = try zpu.fft.fft1d(mixed_signal);
    defer mixed_fft.deinit();
    
    // Apply simple low-pass filter: zero out high frequencies
    var filtered_fft_data: [32]f64 = undefined; // 16 complex = 32 reals
    for (0..n_samples) |i| {
        const re = mixed_fft.readF64(2 * i);
        const im = mixed_fft.readF64(2 * i + 1);
        
        // Keep only frequencies 0, 1, 2, 3 (and their mirrors)
        if (i <= 3 or i >= n_samples - 3) {
            filtered_fft_data[2 * i] = re;
            filtered_fft_data[2 * i + 1] = im;
        } else {
            filtered_fft_data[2 * i] = 0.0;
            filtered_fft_data[2 * i + 1] = 0.0;
        }
    }
    
    const filtered_fft = try zpu.Tensor.fromArray(f64, &filtered_fft_data, try zpu.Shape.init(&.{32}));
    defer filtered_fft.deinit();
    
    const filtered_signal = try zpu.fft.ifft1d(filtered_fft);
    defer filtered_signal.deinit();
    
    std.debug.print("After low-pass filter (keeping freq <= 3):\n", .{});
    std.debug.print("First 8 samples of filtered signal:\n", .{});
    for (0..8) |i| {
        const re = filtered_signal.readF64(2 * i);
        std.debug.print("  [{d}] = {d:.3}\n", .{ i, re });
    }
    
    std.debug.print("\n=== Demo Complete ===\n", .{});
    std.debug.print("\nKey observations:\n", .{});
    std.debug.print("- FFT transforms time-domain signals to frequency domain\n", .{});
    std.debug.print("- Peak in magnitude spectrum indicates dominant frequency\n", .{});
    std.debug.print("- IFFT perfectly reconstructs the original signal\n", .{});
    std.debug.print("- Frequency filtering is easy in the frequency domain\n", .{});
}
