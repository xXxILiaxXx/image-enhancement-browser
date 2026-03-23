import * as ort from "onnxruntime-web";

export function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => resolve({ img, url });
    img.onerror = reject;
    img.src = url;
  });
}

export function createPreviewCanvas(image, width = 128, height = 128) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, width, height);

  return canvas;
}

export function canvasToInputTensor(canvas) {
  const ctx = canvas.getContext("2d");
  const { data, width, height } = ctx.getImageData(
    0,
    0,
    canvas.width,
    canvas.height
  );

  const floatData = new Float32Array(1 * 3 * width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = (y * width + x) * 4;

      const r = data[pixelIndex] / 255;
      const g = data[pixelIndex + 1] / 255;
      const b = data[pixelIndex + 2] / 255;

      const base = y * width + x;
      floatData[base] = r;
      floatData[width * height + base] = g;
      floatData[2 * width * height + base] = b;
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, height, width]);
}