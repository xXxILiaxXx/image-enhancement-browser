import { runModel } from "../../lib/onnx/onnxModel";
import {
  loadImageFromFile,
  createPreviewCanvas,
  canvasToInputTensor,
} from "../../lib/image/imageUtils";
import { applyCorrectionsToImage } from "../../lib/image/correctionUtils";

export async function enhancePhoto(file) {
  const { img, url } = await loadImageFromFile(file);

  const previewCanvas = createPreviewCanvas(img, 128, 128);
  const inputTensor = canvasToInputTensor(previewCanvas);

  const prediction = await runModel(inputTensor);

  const brightness = prediction[0];
  const contrast = prediction[1];
  const saturation = prediction[2];

  const correctedCanvas = applyCorrectionsToImage(
    img,
    brightness,
    contrast,
    saturation
  );

  const correctedDataUrl = correctedCanvas.toDataURL("image/jpeg", 0.95);

  return {
    originalUrl: url,
    correctedUrl: correctedDataUrl,
    prediction: { brightness, contrast, saturation },
  };
}