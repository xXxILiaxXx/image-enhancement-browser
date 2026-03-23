import * as ort from "onnxruntime-web";

let session = null;

export async function getOnnxSession() {
  if (session) return session;

  session = await ort.InferenceSession.create("/model/cnn_regressor.onnx", {
    executionProviders: ["wasm"],
  });

  return session;
}

export async function runModel(inputTensor) {
  const sess = await getOnnxSession();
  const feeds = { input: inputTensor };
  const results = await sess.run(feeds);
  return results.output.data;
}