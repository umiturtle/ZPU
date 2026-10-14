# Images and filters (`zgf`)

## Usage

```zig
const zgf = @import("zgf");

var img = try zgf.Image.init(alloc, 256, 256, 3);   // width, height, channels
defer img.deinit();

img.fill(.{ .r = 0.05, .g = 0.08, .b = 0.12 });
img.verticalGradient(.{ .r = 1, .g = 1, .b = 1 }, .{ .r = 0, .g = 0, .b = 0.4 });
img.drawRect(16, 16, 64, 64, .{ .r = 1, .g = 0.8, .b = 0 });
img.drawCircle(180, 128, 48, .{ .r = 0.9, .g = 0.2, .b = 0.3 });
img.drawLine(0, 0, 255, 255, .{ .r = 1, .g = 1, .b = 1 });
img.drawLineThick(0, 255, 255, 0, 4, .{ .r = 0.2, .g = 1, .b = 0.6 });

const p = img.getPixel(10, 20);

// filters return tensors, so they compose with the rest of zpu
const soft = try img.blur(3);            // separable Gaussian, sigma = radius/2
const edges = try img.sobel();
const crisp = try img.sharpen(1, 1.5);
const bright = try img.brightness(1.2);
const punchy = try img.contrast(1.4);

// 8 bit sRGB and binary PPM/PGM, both directions
const bytes = try img.toPixels(alloc);
defer alloc.free(bytes);
try img.savePpm(init.io, "poster.ppm");
var loaded = try zgf.loadPpm(alloc, init.io, "poster.ppm");
defer loaded.deinit();
```

## Implementation intent

* **An image is a tensor.** `Image` wraps a `zpu.Tensor` shaped `[H, W, C]` with
  values in `0..1`, so every filter is an ordinary tensor expression: it can be
  differentiated, `vmap`ed or `jit`ed like any other function of the pixel data.
  That is the whole reason `zgf` starts from `zpu` rather than from a byte buffer.

* **Filters are composed, not hand-written.** `blur` builds a normalised 1-D
  Gaussian kernel (`gaussianKernel`, sigma = radius/2) and applies it twice,
  along y and along x, by padding and slicing windows and multiplying by the
  kernel taps — a separable convolution expressed in ops. `sobel` convolves with
  the two 3×3 operators and returns the gradient magnitude.

* **The image survives the scope that created it.** `Image.init` calls
  `zpu.untrack` on its tensor, so `img.deinit()` is the single owner regardless of
  the scope in which the image was created. Filter temporaries go through
  `zpu.releaseTemp`, which is a no-op inside a scope and takes the memory back
  outside one.

* **Colour is explicit.** `Color` is an `f32` RGBA value; mono images are written
  with the luminance of the colour, RGB images drop alpha. `toPixels`/`fromPixels`
  convert to and from 8 bit sRGB, and `toPpm`/`savePpm`/`loadPpm` speak the
  simplest binary formats that need no dependency at all (P5/P6).

* **Nothing beyond the standard library.** Writing an image is
  `std.Io.Dir.cwd().createFile(io, path, .{})` plus a header printed with
  `std.fmt`; reading it is one `readPositionalAll` and a tokenised header. PNG
  (which needs zlib) is on the milestone list, not in the tree yet.

## Notes

* Drawing is not anti-aliased: pixels are set or not set. `drawLine` is one pixel
  wide, `drawLineThick` stamps a square brush along the same Bresenham path.
* Out-of-range coordinates are ignored rather than clamped, so drawing partially
  off-canvas is safe.
* Filters return tensors, not images: wrap the result back with
  `Image.fromPixels`-style construction only if you intend to keep drawing.

## Tests

The six tests in `src/zgf.zig` check image construction and pixel access, the 8
bit round trip, the exact bytes of a tiny P6 file, that filters behave (a constant
image blurs to itself, an edge shows up in the Sobel magnitude, brightness and
contrast clamp), the drawing primitives, and that the Gaussian kernel is
normalised and symmetric. All of them run under `std.testing.allocator`.
