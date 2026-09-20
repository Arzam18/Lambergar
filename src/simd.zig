const std = @import("std");
const builtin = @import("builtin");

/// Native SIMD register width in bytes based on target architecture
pub const simd_bytes: usize = blk: {
    if (builtin.cpu.arch.isX86()) {
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) break :blk 64;
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) break :blk 32;
        break :blk 16;
    } else if (builtin.cpu.arch.isAARCH64() or builtin.cpu.arch.isARM()) {
        // ARM NEON registers are 128-bit (16 bytes)
        break :blk 16;
    } else {
        break :blk 16;
    }
};

// Vector element lengths for standard SIMD types
pub const VecI16Len = simd_bytes / @sizeOf(i16); // 8 for NEON, 32 for AVX-512
pub const VecI8Len  = simd_bytes / @sizeOf(i8);  // 16 for NEON, 64 for AVX-512
pub const VecI32Len = simd_bytes / @sizeOf(i32); // 4 for NEON, 16 for AVX-512

pub const VecI16 = @Vector(VecI16Len, i16);
pub const VecI8  = @Vector(VecI8Len, i8);
pub const VecI32 = @Vector(VecI32Len, i32);

/// Vectorized vector addition: dst[i] = a[i] + b[i]
/// On ARM, compiles down to `vaddq_s16` (128-bit NEON add)
pub inline fn addI16(dst: []i16, a: []const i16, b: []const i16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const chunks = dst.len / VecI16Len;

    var i: usize = 0;
    while (i < chunks) : (i += 1) {
        const idx = i * VecI16Len;
        const va: VecI16 = a[idx..][0..VecI16Len].*;
        const vb: VecI16 = b[idx..][0..VecI16Len].*;
        dst[idx..][0..VecI16Len].* = va +% vb;
    }

    // Scalar tail processing
    var tail = chunks * VecI16Len;
    while (tail < dst.len) : (tail += 1) {
        dst[tail] = a[tail] +% b[tail];
    }
}

/// Vectorized vector subtraction: dst[i] = a[i] - b[i]
/// On ARM, compiles down to `vsubq_s16`
pub inline fn subI16(dst: []i16, a: []const i16, b: []const i16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const chunks = dst.len / VecI16Len;

    var i: usize = 0;
    while (i < chunks) : (i += 1) {
        const idx = i * VecI16Len;
        const va: VecI16 = a[idx..][0..VecI16Len].*;
        const vb: VecI16 = b[idx..][0..VecI16Len].*;
        dst[idx..][0..VecI16Len].* = va -% vb;
    }

    var tail = chunks * VecI16Len;
    while (tail < dst.len) : (tail += 1) {
        dst[tail] = a[tail] -% b[tail];
    }
}

/// Clipped ReLU Activation + Quantization for NNUE
/// Clamps input to [0, max_val] and right-shifts by `shift`
pub inline fn clippedReluQuantize(
    dst: []i8,
    src: []const i16,
    comptime max_val: i16,
    comptime shift: u3,
) void {
    const zero_vec: VecI16 = @splat(0);
    const max_vec: VecI16 = @splat(max_val);
    const chunks = src.len / VecI16Len;

    var i: usize = 0;
    while (i < chunks) : (i += 1) {
        const src_idx = i * VecI16Len;
        const dst_idx = i * VecI16Len;

        const v: VecI16 = src[src_idx..][0..VecI16Len].*;
        const clamped = @min(@max(v, zero_vec), max_vec);
        const shifted = clamped >> @splat(shift);

        // Narrow i16 to i8
        var tmp: [VecI16Len]i8 = undefined;
        inline for (0..VecI16Len) |j| {
            tmp[j] = @truncate(shifted[j]);
        }
        dst[dst_idx..][0..VecI16Len].* = tmp;
    }
}

/// Multiplies accumulator values with weights and reduces to scalar i32.
/// Translates to NEON pairwise addition or `sdot` instructions on AArch64.
pub fn dotProductI8I8(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len);
    const chunks = a.len / VecI8Len;
    var acc: VecI32 = @splat(0);

    var i: usize = 0;
    while (i < chunks) : (i += 1) {
        const idx = i * VecI8Len;
        const va: VecI8 = a[idx..][0..VecI8Len].*;
        const vb: VecI8 = b[idx..][0..VecI8Len].*;

        // Widening multiplication i8 -> i32 to prevent overflow
        const va_32: VecI32 = @alignCast(va);
        const vb_32: VecI32 = @alignCast(vb);

        acc += va_32 * vb_32;
    }

    var result: i32 = @reduce(.add, acc);

    // Scalar tail
    var tail = chunks * VecI8Len;
    while (tail < a.len) : (tail += 1) {
        result += @as(i32, a[tail]) * @as(i32, b[tail]);
    }

    return result;
}
