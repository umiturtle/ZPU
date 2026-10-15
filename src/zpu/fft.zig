//! Fast Fourier Transform operations
//!
//! Provides 1D and 2D FFT operations similar to jax.numpy.fft.
//! Uses the Cooley-Tukey radix-2 algorithm for power-of-2 sizes.

const std = @import("std");
const tmod = @import("tensor.zig");
const ops = @import("ops.zig");

pub const Tensor = tmod.Tensor;
pub const DType = tmod.DType;
pub const Shape = tmod.Shape;
pub const Error = tmod.Error;
pub const releaseTemp = tmod.releaseTemp;

/// Complex number representation for FFT
pub const Complex = struct {
    re: f64,
    im: f64,

    pub fn add(a: Complex, b: Complex) Complex {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }

    pub fn sub(a: Complex, b: Complex) Complex {
        return .{ .re = a.re - b.re, .im = a.im - b.im };
    }

    pub fn mul(a: Complex, b: Complex) Complex {
        return .{
            .re = a.re * b.re - a.im * b.im,
            .im = a.re * b.im + a.im * b.re,
        };
    }

    pub fn conj(a: Complex) Complex {
        return .{ .re = a.re, .im = -a.im };
    }

    pub fn abs(a: Complex) f64 {
        return @sqrt(a.re * a.re + a.im * a.im);
    }

    pub fn fromPolar(r: f64, theta: f64) Complex {
        return .{
            .re = r * @cos(theta),
            .im = r * @sin(theta),
        };
    }
};

/// 1D Fast Fourier Transform (Cooley-Tukey algorithm)
/// Input: 1D tensor of length n (must be power of 2)
/// Output: 1D tensor of complex values (stored as interleaved re,im)
pub fn fft1d(x: Tensor) Error!Tensor {
    if (x.rank() != 1) return Error.ShapeMismatch;
    const n = x.shape.dims[0];
    
    // Check if n is power of 2
    if (n == 0 or (n & (n - 1)) != 0) {
        return Error.NotSupported; // Non-power-of-2 not yet supported
    }
    
    // Output: n complex numbers = 2n real numbers
    var out = try Tensor.empty(try Shape.init(&.{n * 2}), .f64);
    
    // Extract input as complex (imaginary part = 0)
    var input = try std.heap.c_allocator.alloc(Complex, n);
    defer std.heap.c_allocator.free(input);
    
    for (0..n) |i| {
        input[i] = .{ .re = x.readF64(i), .im = 0.0 };
    }
    
    // Perform FFT using Cooley-Tukey
    var output = try std.heap.c_allocator.alloc(Complex, n);
    defer std.heap.c_allocator.free(output);
    
    fftRecursive(input, output, n);
    
    // Store result as interleaved real/imaginary
    for (0..n) |i| {
        out.writeF64(2 * i, output[i].re);
        out.writeF64(2 * i + 1, output[i].im);
    }
    
    return out;
}

/// Inverse 1D FFT
pub fn ifft1d(x: Tensor) Error!Tensor {
    if (x.rank() != 1) return Error.ShapeMismatch;
    const n = x.shape.dims[0];
    if (n % 2 != 0) return Error.ShapeMismatch;
    const num_complex = n / 2;
    
    // Check if num_complex is power of 2
    if (num_complex == 0 or (num_complex & (num_complex - 1)) != 0) {
        return Error.NotSupported;
    }
    
    // Extract complex input
    var input = try std.heap.c_allocator.alloc(Complex, num_complex);
    defer std.heap.c_allocator.free(input);
    
    for (0..num_complex) |i| {
        input[i] = .{
            .re = x.readF64(2 * i),
            .im = x.readF64(2 * i + 1),
        };
    }
    
    // Conjugate input
    for (input) |*c| {
        c.im = -c.im;
    }
    
    // Perform FFT (IFFT via conjugation trick)
    var output = try std.heap.c_allocator.alloc(Complex, num_complex);
    defer std.heap.c_allocator.free(output);
    
    fftRecursive(input, output, num_complex);
    
    // Conjugate and scale
    var out = try Tensor.empty(try Shape.init(&.{num_complex * 2}), .f64);
    for (0..num_complex) |i| {
        out.writeF64(2 * i, output[i].re / @as(f64, @floatFromInt(num_complex)));
        out.writeF64(2 * i + 1, -output[i].im / @as(f64, @floatFromInt(num_complex)));
    }
    
    return out;
}

/// Recursive FFT implementation (Cooley-Tukey)
fn fftRecursive(input: []Complex, output: []Complex, n: usize) void {
    if (n == 1) {
        output[0] = input[0];
        return;
    }
    
    const half = n / 2;
    
    // Allocate temporary arrays
    var even_in = std.heap.c_allocator.alloc(Complex, half) catch unreachable;
    defer std.heap.c_allocator.free(even_in);
    var odd_in = std.heap.c_allocator.alloc(Complex, half) catch unreachable;
    defer std.heap.c_allocator.free(odd_in);
    var even_out = std.heap.c_allocator.alloc(Complex, half) catch unreachable;
    defer std.heap.c_allocator.free(even_out);
    var odd_out = std.heap.c_allocator.alloc(Complex, half) catch unreachable;
    defer std.heap.c_allocator.free(odd_out);
    
    // Split into even and odd
    for (0..half) |i| {
        even_in[i] = input[2 * i];
        odd_in[i] = input[2 * i + 1];
    }
    
    // Recursive FFT on each half
    fftRecursive(even_in, even_out, half);
    fftRecursive(odd_in, odd_out, half);
    
    // Combine with twiddle factors
    for (0..half) |k| {
        const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n));
        const twiddle = Complex.fromPolar(1.0, angle);
        const t = twiddle.mul(odd_out[k]);
        output[k] = even_out[k].add(t);
        output[k + half] = even_out[k].sub(t);
    }
}

/// Compute magnitude spectrum (absolute values of FFT)
pub fn fftMag(x: Tensor) Error!Tensor {
    const fft_result = try fft1d(x);
    defer fft_result.deinit();
    
    const n = x.shape.dims[0];
    var out = try Tensor.empty(try Shape.init(&.{n}), .f64);
    
    for (0..n) |i| {
        const re = fft_result.readF64(2 * i);
        const im = fft_result.readF64(2 * i + 1);
        out.writeF64(i, @sqrt(re * re + im * im));
    }
    
    return out;
}

/// Compute power spectrum (squared magnitudes)
pub fn fftPower(x: Tensor) Error!Tensor {
    const fft_result = try fft1d(x);
    defer fft_result.deinit();
    
    const n = x.shape.dims[0];
    var out = try Tensor.empty(try Shape.init(&.{n}), .f64);
    
    for (0..n) |i| {
        const re = fft_result.readF64(2 * i);
        const im = fft_result.readF64(2 * i + 1);
        out.writeF64(i, re * re + im * im);
    }
    
    return out;
}

/// 2D FFT (applies 1D FFT to rows, then to columns)
pub fn fft2d(x: Tensor) Error!Tensor {
    if (x.rank() != 2) return Error.ShapeMismatch;
    const rows = x.shape.dims[0];
    const cols = x.shape.dims[1];
    
    // Check if dimensions are powers of 2
    if (rows == 0 or (rows & (rows - 1)) != 0 or cols == 0 or (cols & (cols - 1)) != 0) {
        return Error.NotSupported;
    }
    
    // FFT along rows
    var temp = try Tensor.empty(try Shape.init(&.{ rows * cols * 2 }), .f64);
    
    for (0..rows) |r| {
        // Extract row
        var row = try Tensor.empty(try Shape.init(&.{cols}), x.dt);
        for (0..cols) |c| {
            row.writeF64(c, x.readF64(r * cols + c));
        }
        
        // FFT the row
        const row_fft = try fft1d(row);
        defer row_fft.deinit();
        row.deinit();
        
        // Store result
        for (0..cols * 2) |i| {
            temp.writeF64(r * cols * 2 + i, row_fft.readF64(i));
        }
    }
    
    // FFT along columns
    var out = try Tensor.empty(try Shape.init(&.{ rows * cols * 2 }), .f64);
    
    for (0..cols) |c| {
        // Extract column (complex values)
        var col_re = try Tensor.empty(try Shape.init(&.{rows}), .f64);
        var col_im = try Tensor.empty(try Shape.init(&.{rows}), .f64);
        
        for (0..rows) |r| {
            col_re.writeF64(r, temp.readF64(r * cols * 2 + 2 * c));
            col_im.writeF64(r, temp.readF64(r * cols * 2 + 2 * c + 1));
        }
        
        // FFT real and imaginary parts separately
        const col_fft_re = try fft1d(col_re);
        defer col_fft_re.deinit();
        const col_fft_im = try fft1d(col_im);
        defer col_fft_im.deinit();
        col_re.deinit();
        col_im.deinit();
        
        // Combine (FFT(a+bi) = FFT(a) + i*FFT(b))
        for (0..rows) |r| {
            // Real part: re_fft_re - im_fft_im
            out.writeF64(r * cols * 2 + 2 * c, 
                col_fft_re.readF64(2 * r) - col_fft_im.readF64(2 * r + 1));
            // Imaginary part: re_fft_im + im_fft_re
            out.writeF64(r * cols * 2 + 2 * c + 1,
                col_fft_re.readF64(2 * r + 1) + col_fft_im.readF64(2 * r));
        }
    }
    
    temp.deinit();
    
    // Reshape to 2D complex
    return Tensor.empty(try Shape.init(&.{ rows, cols * 2 }), .f64);
}

test "1D FFT basic" {
    const alloc = std.testing.allocator;
    
    // Simple test: FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
    const x = try Tensor.fromArray(f64, &.{ 1.0, 0.0, 0.0, 0.0 }, try Shape.init(&.{4}));
    defer x.deinit();
    
    const result = try fft1d(x);
    defer result.deinit();
    
    // Check that all magnitudes are 1
    for (0..4) |i| {
        const re = result.readF64(2 * i);
        const im = result.readF64(2 * i + 1);
        const mag = @sqrt(re * re + im * im);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), mag, 1e-10);
    }
    
    _ = alloc;
}

test "FFT inverse" {
    // Test that IFFT(FFT(x)) = x
    const x = try Tensor.fromArray(f64, &.{ 1.0, 2.0, 3.0, 4.0 }, try Shape.init(&.{4}));
    defer x.deinit();
    
    const fft_result = try fft1d(x);
    defer fft_result.deinit();
    
    const recovered = try ifft1d(fft_result);
    defer recovered.deinit();
    
    // Check reconstruction
    for (0..4) |i| {
        const re = recovered.readF64(2 * i);
        try std.testing.expectApproxEqAbs(@as(f64, @floatFromInt(i + 1)), re, 1e-10);
    }
}
