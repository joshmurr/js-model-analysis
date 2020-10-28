import * as tf from '@tensorflow/tfjs';
import OutputGL from './outputGL.js';

const MODELS = {
  // 'Flowers_Uncompressed_H5' : 'models/desaturate2flower/model.json',
  // 'webcam2flower_uncompressed' : 'models/webcam2flower/uncompressed/model.json',
  // 'webcam2flower_uint8_compressed' : 'models/webcam2flower/uint8_compressed/model.json',
  flowers_256_64: 'models/greyscale2flowers/uncompressed/model.json',
  flowers_256_8: 'models/flowers_256_8/model.json',
  clouds_256_4: 'models/clouds_256_4/model.json',
  // 'greyscale2clouds' : 'models/greyscale2clouds/model.json',
  // 'greyscale2forest' : 'models/greyscale2forest/model.json',
  //greyscale2waves: 'models/waves_128/uncompressed/model.json',
  'User Upload': 'NULL',
};

// const DEFAULT_MODEL = 'Flowers_Uncompressed_H5';
const DEFAULT_MODEL = 'greyscale2flowers';
const DEFAULT_SIZE = 256;
let IMAGE_SIZE = DEFAULT_SIZE;
let MODEL_INPUT_SHAPE;

let videoPlaying = false;
let MODEL_LOADED = false;

let userModel = {
  json: null,
  weights: null,
};

// Text Outputs:
const statusElement = document.getElementById('status');
const status = (msg, state) => {
  statusElement.classList = '';
  statusElement.innerText = msg;
  statusElement.classList.add(state);
};
const errorsElement = document.getElementById('errors');
const errors = (msg) => (errorsElement.innerText = msg);

// Visual I/O
const img = document.getElementById('test_img');
const vid = document.getElementById('test_vid');
const imgBtn = document.getElementById('image_button');
const vidBtn = document.getElementById('video_button');
const camBtn = document.getElementById('webcam_button');
const modelBtn = document.getElementById('model_button');
const outputImage = document.getElementById('output_image_canvas');
const outputCV2Image = document.getElementById('cv2_image_canvas');
const outputCV2Video = document.getElementById('cv2_video_canvas');
const outputVideoTF = document.getElementById('output_video_tf');
const outputVideoGL = document.getElementById('output_video_gl');
const modelSelect = document.getElementById('model_select');

// WebGL
const outputGL = new OutputGL(outputVideoGL);

// STATS
const backendElement = document.getElementById('backend');
const model_upload_time_text = document.getElementById('model_upload_time');

const image_cv2_preprocess = document.getElementById('image_cv2_preprocess');
const image_model_inference = document.getElementById('image_model_inference');
const image_to_canvas = document.getElementById('image_to_canvas');

const video_cv2_preprocess = document.getElementById('video_cv2_preprocess');
const video_model_inference = document.getElementById('video_model_inference');
const video_to_canvas = document.getElementById('video_to_canvas');

const fps_text = document.getElementById('fps');
let inference_count = 0;
let total_inference_time = 0;
let average_inference_time = 0;

// Stats tf.Memory()
const tfMemoryDOM = {
  tfnumBytes: document.getElementById('tfMemNumBytes'),
  tfnumBytesInGPU: document.getElementById('tfMemNumBytesInGPU'),
  tfnumBytesInGPUAllocated: document.getElementById(
    'tfMemNumBytesInGPUAllocated'
  ),
  tfnumBytesInGPUFree: document.getElementById('tfMemNumBytesInGPUFree'),
  tfnumDataBuffers: document.getElementById('tfMemNumDataBuffers'),
  tfnumTensors: document.getElementById('tfMemNumTensors'),
  tfunreliable: document.getElementById('tfMemUnreliable'),
};

// CHECKBOXES
const desaturateBox = document.getElementById('desaturate_checkbox');
const greyscaleBox = document.getElementById('greyscale_checkbox');
const downUpscaleBox = document.getElementById('downupscale_checkbox');
const brightnessBox = document.getElementById('brightness_checkbox');
const invertBox = document.getElementById('invert_checkbox');
const glBox = document.getElementById('gl_checkbox');

function init() {
  for (let model in MODELS) {
    let opt = document.createElement('option');
    let text = document.createTextNode(model);
    opt.value = model;
    opt.appendChild(text);
    modelSelect.appendChild(opt);
  }
}

let model;
async function loadModel(modelID) {
  if (model) {
    status('Clearing previous model...', 'loading');
    model.dispose();
  }

  status('Loading model...', 'loading');

  const startTime = performance.now();
  try {
    if (modelID === 'User Upload') {
      const load = tf.io.browserFiles([userModel.json, ...userModel.weights]);
      model = await tf.loadGraphModel(load, { strict: true });
    } else {
      model = await tf.loadGraphModel(MODELS[modelID], { strict: true });
    }
  } catch (err) {
    status('Error loading model!', 'bad');
    errors(err);
    console.log(err);
    return;
  }

  // parseJSON() populates DOM elements for GUI output and
  // returns the input tensor shape.
  MODEL_INPUT_SHAPE = parseJSON(model.artifacts).map((dim) =>
    Math.abs(dim.size)
  );

  if (MODEL_INPUT_SHAPE[1] !== DEFAULT_SIZE) {
    IMAGE_SIZE = MODEL_INPUT_SHAPE[1];
    resizeCanvases(MODEL_INPUT_SHAPE);
    const imageFilesElement = document.getElementById('image-file');
    if ('createEvent' in document) {
      const evt = document.createEvent('HTMLEvents');
      evt.initEvent('change', false, true);
      imageFilesElement.dispatchEvent(evt);
    } else {
      imageFilesElement.fireEvent('onchange');
    }
  }

  model.predict(tf.zeros(MODEL_INPUT_SHAPE)).dispose();

  const totalTime = performance.now() - startTime;
  model_upload_time_text.innerText = totalTime;

  status('Model Loaded Successfully.', 'good');
  MODEL_LOADED = true;

  backendElement.innerText = tf.getBackend();

  // Populate DOM Output with tfMemory details:
  const tfMemoryOutput = tf.memory();
  for (const item in tfMemoryOutput) {
    const selector = 'tf' + item;
    if (item.includes('Bytes'))
      tfMemoryDOM[selector].innerText = tfMemoryOutput[item] * 1e-6;
    else tfMemoryDOM[selector].innerText = tfMemoryOutput[item];
  }
}

function resizeCanvases(shape) {
  for (const canvas of document.getElementsByTagName('canvas')) {
    canvas.width = shape[1];
    canvas.height = shape[2];
    canvas.style.width = '256px';
    canvas.style.height = '256px';
  }
  outputGL.deleteTexture();
  outputGL.createTexture(outputVideoGL);
}

async function predict(imgElement, outputCanvas, gl = false) {
  if (!MODEL_LOADED) {
    init(DEFAULT_MODEL);
  } else {
    status('Inferencing...', 'loading');
    const logits = tf.tidy(() => {
      const img = tf.browser
        .fromPixels(imgElement, MODEL_INPUT_SHAPE[3])
        .toFloat();

      const offset = tf.scalar(127.5);
      const normalized = img.sub(offset).div(offset);

      const batched = normalized.reshape(MODEL_INPUT_SHAPE);
      return model.predict(batched);
    });

    const output = await postProcessTF(logits);

    status('Finished Inferencing.', 'good');

    if (gl) {
      let data = await output.data();
      outputGL.draw(data);
    } else {
      tf.browser.toPixels(output, outputCanvas);
    }
  }
}

async function postProcessTF(logits) {
  return tf.tidy(() => {
    const scale = tf.scalar(0.5);
    const squeezed = logits.squeeze().mul(scale).add(scale);
    const resized = tf.image.resizeBilinear(squeezed, [IMAGE_SIZE, IMAGE_SIZE]);
    return resized;
  });
}

const imageFilesElement = document.getElementById('image-file');
imageFilesElement.addEventListener('change', (evt) => {
  let files = evt.target.files;
  for (let i = 0, f; (f = files[i]); i++) {
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = (e) => {
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
    };
    reader.readAsDataURL(f);
  }
});

const videoFilesElement = document.getElementById('video-file');
videoFilesElement.addEventListener('change', (evt) => {
  let files = evt.target.files;
  for (let i = 0, f; (f = files[i]); i++) {
    if (!f.type.match('video.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = (e) => {
      vid.src = e.target.result;
      vid.onload = () => processVideo(vid);
    };
    reader.readAsDataURL(f);
  }
});

const jsonFileElement = document.getElementById('json-file');
jsonFileElement.addEventListener('change', (evt) => {
  let files = evt.target.files;
  if (files.length > 1) {
    error('There should only be one JSON file.');
    return;
  }
  for (let i = 0, f; (f = files[i]); i++) {
    if (!f.type === 'application/json') {
      error('Filetype should be JSON!');
      continue;
    }
  }

  userModel.json = files[0];

  modelSelect.value = 'User Upload';
  status('Successfully loaded model JSON.', 'good');
});

function parseJSON(json) {
  const jsonDOM = {
    convertedBy: document.getElementById('convertedBy'),
    format: document.getElementById('format'),
    generatedBy: document.getElementById('generatedBy'),
    inputTensorname: document.getElementById('inputTensorName'),
    inputTensordtype: document.getElementById('inputTensorType'),
    inputTensortensorShape: document.getElementById('inputTensorShape'),
    outputTensorname: document.getElementById('outputTensorName'),
    outputTensordtype: document.getElementById('outputTensorType'),
    outputTensortensorShape: document.getElementById('outputTensorShape'),
  };

  const modelTopology = json['modelTopology'];

  for (const item in json) {
    if (jsonDOM.hasOwnProperty(item)) {
      jsonDOM[item].innerText = json[item];
    }
  }

  const inputs = json.userDefinedMetadata.signature.inputs;
  let input_shape;
  for (const item in inputs) {
    for (const input in inputs[item]) {
      if (input === 'tensorShape') {
        input_shape = inputs[item][input]['dim'];
        let st = '';
        for (const d in input_shape) {
          st = st.concat(input_shape[d]['size'], ', ');
        }
        jsonDOM['inputTensortensorShape'].innerText = st;
      } else {
        jsonDOM['inputTensor' + input].innerText = inputs[item][input];
      }
    }
  }

  const outputs = json.userDefinedMetadata.signature.outputs;
  for (const item in outputs) {
    for (const output in outputs[item]) {
      if (output === 'tensorShape') {
        let st = '';
        for (const d in outputs[item][output]['dim']) {
          st = st.concat(outputs[item][output]['dim'][d]['size'], ', ');
        }
        jsonDOM['outputTensortensorShape'].innerText = st;
      } else {
        jsonDOM['outputTensor' + output].innerText = outputs[item][output];
      }
    }
  }
  return input_shape;
}

const weightsFilesElement = document.getElementById('weights-files');
weightsFilesElement.addEventListener('change', (evt) => {
  let files = evt.target.files;
  for (let i = 0, f; (f = files[i]); i++) {
    if (!f.type === 'application/octet-stream') {
      error('Wrong Weights filetype!');
      continue;
    }
  }
  userModel.weights = files;
  modelSelect.value = 'User Upload';
  status('Successfully loaded model weights.', 'good');
});

function processVideo(videoElement) {
  let cap = new cv.VideoCapture(videoElement);
  let frame = new cv.Mat(videoElement.height, videoElement.width, cv.CV_8UC4); // CORRECT

  let dst = new cv.Mat();

  let frames = 0;
  let numberOfDrops = 0;
  let lowestFrameRate = -1;
  let time;
  let times = [];
  let fps;
  function stream() {
    try {
      if (!videoPlaying) {
        dst.delete();
        frame.delete();
        return;
      }
      const begin = performance.now();

      while (times.length > 0 && times[0] <= begin - 1000) {
        times.shift();
      }
      times.push(begin);

      fps = times.length;

      cap.read(frame);
      preprocessImageCV2(
        frame,
        outputCV2Video,
        downUpscaleBox.checked,
        greyscaleBox.checked,
        brightnessBox.checked,
        desaturateBox.checked,
        invertBox.checked
      );

      const preprocessed = performance.now();

      predict(outputCV2Video, outputVideoTF, glBox.checked);

      const inferenced = performance.now();

      video_cv2_preprocess.innerText = preprocessed - begin;
      video_model_inference.innerText = inferenced - preprocessed;
      fps_text.innerText = fps;

      requestAnimationFrame(stream); // Force FPS
    } catch (err) {
      console.log(err);
    }
  }
  requestAnimationFrame(stream);
}

function initWebcam(video) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.log('enumerateDevices() not supported!');
    return;
  }
  navigator.getUserMedia =
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia;
  if (navigator.mediaDevices) {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function (err) {
        console.log('An error occurred loading the webcam! ' + err);
      });
  } else {
    navigator
      .getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function (err) {
        console.log('An error occurred loading the webcam! ' + err);
      });
  }
}

function preprocessImageCV2(
  imgElement,
  outputCanvas,
  downUpscale,
  greyscale,
  brightness,
  desaturate,
  invert
) {
  let output;
  let width, height;
  if (imgElement instanceof HTMLElement) {
    width = imgElement.width;
    height = imgElement.height;
    output = cv.imread(imgElement);
  } else {
    width = imgElement.cols;
    height = imgElement.rows;
    output = imgElement.clone();
  }

  if (width !== IMAGE_SIZE || height !== IMAGE_SIZE) {
    const offset_x = Math.floor((width - IMAGE_SIZE) / 2);
    const offset_y = Math.floor((height - IMAGE_SIZE) / 2);
    const mask = new cv.Rect(offset_x, offset_y, IMAGE_SIZE, IMAGE_SIZE);

    output = output.roi(mask);
  }

  const smallSize = Math.floor(IMAGE_SIZE / 24);
  const dsize = new cv.Size(smallSize, smallSize);
  const usize = new cv.Size(IMAGE_SIZE, IMAGE_SIZE);

  if (greyscale) {
    cv.cvtColor(output, output, cv.COLOR_RGB2GRAY, 0);
  }
  if (invert) {
    cv.bitwise_not(output, output);
  }
  if (desaturate) {
    cv.threshold(output, output, 150, 200, cv.THRESH_TRUNC); // Desaturate
  }
  if (brightness) {
    cv.convertScaleAbs(output, output, 2, 75);
  }
  if (downUpscale) {
    cv.resize(output, output, dsize, 0, 0, cv.INTER_AREA); // Downscale
    cv.resize(output, output, usize, 0, 0, cv.INTER_CUBIC); // Upscale
  }

  cv.imshow(outputCanvas, output);
  output.delete();
}

modelBtn.addEventListener('click', (e) => {
  loadModel(modelSelect.options[modelSelect.selectedIndex].value);
});

vid.onplay = () => {
  videoPlaying = true;
  processVideo(vid);
};

vid.onended = () => {
  videoPlaying = false;
};
vid.onpause = () => {
  videoPlaying = false;
};

imgBtn.addEventListener('click', (e) => {
  if (img.complete && img.naturalHeight !== 0) {
    const processImg_start = performance.now();

    preprocessImageCV2(
      img,
      outputCV2Image,
      downUpscaleBox.checked,
      greyscaleBox.checked,
      brightnessBox.checked,
      desaturateBox.checked,
      invertBox.checked
    );

    const processImg_end = performance.now();

    predict(outputCV2Image, outputImage);

    const infImg_end = performance.now() - processImg_end;
    image_cv2_preprocess.innerText = processImg_end - processImg_start;
    image_model_inference.innerText = infImg_end;
  } else {
    console.warn('Image not yet loaded.');
  }
});
vidBtn.addEventListener('click', (e) => {
  vid.srcObject = null;
  vid.play();
  videoPlaying = true;
  processVideo(vid);
});
camBtn.addEventListener('click', (e) => {
  initWebcam(vid);
  videoPlaying = true;
  processVideo(vid);
});

init();
