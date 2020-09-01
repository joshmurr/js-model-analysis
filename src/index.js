import * as tf from '@tensorflow/tfjs';
//import * as utils from 'utils.js';

const MODELS = {
  'Flowers_Uncompressed_H5' : 'models/h5_graph/model.json',
  'NULL' : 'NULL',
};

const DEFAULT_MODEL = 'Flowers_Uncompressed_H5';
const IMAGE_SIZE = 256;

let videoPlaying = false;
let MODEL_LOADED = false;

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const img = document.getElementById('test_img');
const vid = document.getElementById('test_vid');
const imgBtn = document.getElementById('image_button');
const vidBtn = document.getElementById('video_button');
const modelBtn = document.getElementById('model_button');
const outputImage = document.getElementById('output_image_canvas');
const outputCV2Image = document.getElementById('cv2_image_canvas');
const outputCV2Video = document.getElementById('cv2_video_canvas');
const outputVideo = document.getElementById('output_video_canvas');
let modelSelect = document.getElementById('model_select');

// STATS
const model_upload_time_text = document.getElementById('model_upload_time');
const recent_inference_time_text = document.getElementById('recent_inference_time');
const average_inference_time_text = document.getElementById('average_inference_time');
const image_preprocess_time_text = document.getElementById('image_preprocess_time');
const fps_text = document.getElementById('fps');
let inference_count = 0;
let total_inference_time = 0;
let average_inference_time = 0;

function updateInferenceTime(inferenceTime){
  inference_count++;
  total_inference_time += inferenceTime;
  average_inference_time = total_inference_time/inference_count;

  recent_inference_time_text.innerText = inferenceTime;
  average_inference_time_text.innerText = average_inference_time;
}

function init(){
  for(let model in MODELS){
    let opt = document.createElement('option');
    let text = document.createTextNode(model);
    opt.value = model;
    opt.appendChild(text);
    modelSelect.appendChild(opt);
  }
}

let model;
async function loadModel(modelID) {
  status('Loading model...');
  const startTime = performance.now();
  try{
    model = await tf.loadGraphModel(MODELS[modelID], {strict: true});
  } catch (err){
    status(err.message);
    console.log("ERROR LOADING MODEL");
  }

  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  const totalTime = performance.now() - startTime;
  model_upload_time_text.innerText = totalTime;

  status('Model Loaded Successfully.');
  MODEL_LOADED = true;
};

async function predict(imgElement, outputCanvas) {
  if(!MODEL_LOADED) {
    init(DEFAULT_MODEL);
  } else {
    status('Inferencing...');

    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    const startTime = performance.now();
    let startTime2;
    const logits = tf.tidy(() => {
      const img = tf.browser.fromPixels(imgElement).toFloat();

      // PREPROCESSING
      const scale = 24;
      const downscale = Math.floor(IMAGE_SIZE/scale);
      const downscaled = img.resizeNearestNeighbor([downscale, downscale]);
      const upscaled = downscaled.resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])

      const offset = tf.scalar(127.5);
      const normalized = upscaled.sub(offset).div(offset);

      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      startTime2 = performance.now();
      return model.predict(batched);
    });

    const output = await preprocess(logits);
    const scale = tf.scalar(0.5);
    const fin = output.mul(scale).add(scale);

    const endTime = performance.now();
    const totalTime = endTime - startTime;
    const preprocessTime = endTime - startTime2;
    updateInferenceTime(totalTime);
    image_preprocess_time_text.innerText = preprocessTime;
    status('Finished Inferencing.');

    tf.browser.toPixels(fin, outputCanvas);
  }
}

async function preprocess(logits){
  return tf.tidy(() => {
    const scale = tf.scalar(0.5);
    const squeezed = logits.squeeze().mul(scale).add(scale);
    const resized = tf.image.resizeBilinear(squeezed, [IMAGE_SIZE, IMAGE_SIZE]);

    return resized;
  })
}

const imageFilesElement = document.getElementById('image-file');
imageFilesElement.addEventListener('change', evt => {
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

const videoFilesElement = document.getElementById('video-file');
videoFilesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  for (let i = 0, f; f = files[i]; i++) {
    if (!f.type.match('video.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      vid.src = e.target.result;
      vid.onload = () => processVideo();
    };
    reader.readAsDataURL(f);
  }
});

function processVideo(){
  let cap = new cv.VideoCapture(vid);
  let frame = new cv.Mat(vid.height, vid.width, cv.CV_8UC4); // CORRECT

  const offset_y = Math.floor((vid.height - IMAGE_SIZE) / 2);
  const offset_x = Math.floor((vid.width - IMAGE_SIZE) / 2);
  let dst = new cv.Mat();
  let mask = new cv.Rect(offset_x, offset_y, IMAGE_SIZE, IMAGE_SIZE);

  const FPS = 25;
  let total_delay=0;
  let i=0;
  function stream(){
    try{
      if(!videoPlaying){
        dst.delete();
        frame.delete();
        return;
      }
      let begin = Date.now();
      cap.read(frame);
      dst = frame.roi(mask);
      cv.imshow(outputCV2Video, dst);

      predict(outputCV2Video, outputVideo);

      let delay = 1000/FPS - (Date.now() - begin);
      total_delay+=delay;
      if(i%10==0) fps_text.innerText = Math.floor(total_delay/i);
      i++;
      setTimeout(stream, delay); 
    } catch (err) {
      console.log(err);
    }
  }
 setTimeout(stream, 0); 
}

modelBtn.addEventListener('click', e => {
  loadModel(modelSelect.options[modelSelect.selectedIndex].value)
});

vid.onplay = () => {
  videoPlaying = true;
  processVideo();
}

vid.onended = () => {
  videoPlaying = false;
}
vid.onpause = () => {
  videoPlaying = false;
}

imgBtn.addEventListener('click', e => {
  if (img.complete && img.naturalHeight !== 0) {
    predict(img, outputImage);
  } else {
    img.onload = () => {
      predict(img, outputImage);
    }
  }
});
vidBtn.addEventListener('click', e => {
  vid.play();
  videoPlaying = true;
  processVideo()
});

init();
