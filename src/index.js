import * as tf from '@tensorflow/tfjs';

const MODEL_PATH = 'models/h5_graph/model.json';
const IMAGE_SIZE = 256;

let model;
const demo = async () => {
  status('Loading model...');
  
  model = await tf.loadGraphModel(MODEL_PATH, {strict: true});

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  if (img.complete && img.naturalHeight !== 0) {
    predict(img);
  } else {
    img.onload = () => {
      predict(img);
    }
  }

  document.getElementById('file-container').style.display = '';
};

async function predict(imgElement) {
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    // PREPROCESSING
    const scale = 24;
    const downscale = Math.floor(IMAGE_SIZE/scale);
    const downscaled = img.resizeNearestNeighbor([downscale, downscale]);
    const upscaled = downscaled.resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])

    const offset = tf.scalar(127.5);
    const normalized = upscaled.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    return model.predict(batched);
  });

  const output = await preprocess(logits);
  const scale = tf.scalar(0.5);
  const fin = output.mul(scale).add(scale);

  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  tf.browser.toPixels(fin, outputCanvas);
}

async function preprocess(logits){
  return tf.tidy(() => {
    const scale = tf.scalar(0.5);
    const squeezed = logits.squeeze().mul(scale).add(scale);
    const resized = tf.image.resizeBilinear(squeezed, [IMAGE_SIZE, IMAGE_SIZE]);

    return resized;
  })
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      // let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});


const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const img = document.getElementById('test_img');
const outputCanvas = document.getElementById('output_canvas');

demo();
