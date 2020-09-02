import * as tf from '@tensorflow/tfjs';
import { preprocessImageCV2 } from './image_processing.js';
//import * as utils from 'utils.js';

const MODELS = {
  'Flowers_Uncompressed_H5' : 'models/h5_graph/model.json',
  'User Upload' : 'NULL',
};

const DEFAULT_MODEL = 'Flowers_Uncompressed_H5';
const IMAGE_SIZE = 256;

let videoPlaying = false;
let MODEL_LOADED = false;

let userModel= {
  json: null,
  weights: null,
}

const statusElement = document.getElementById('status');
const status = msg => statusElement.innerText = msg;
const errorsElement = document.getElementById('errors');
const errors = msg => errorsElement.innerText = msg;

const img = document.getElementById('test_img');
const vid = document.getElementById('test_vid');
const imgBtn = document.getElementById('image_button');
const vidBtn = document.getElementById('video_button');
const modelBtn = document.getElementById('model_button');
const outputImage = document.getElementById('output_image_canvas');
const outputCV2Image = document.getElementById('cv2_image_canvas');
const outputCV2Video = document.getElementById('cv2_video_canvas');
const outputVideo = document.getElementById('output_video_canvas');
const modelSelect = document.getElementById('model_select');

// STATS
const model_upload_time_text = document.getElementById('model_upload_time');
const recent_inference_time_text = document.getElementById('recent_inference_time');
const average_inference_time_text = document.getElementById('average_inference_time');
const image_preprocess_time_text = document.getElementById('image_preprocess_time');
const fps_text = document.getElementById('fps');
let inference_count = 0;
let total_inference_time = 0;
let average_inference_time = 0;
// Stats tf.Memory()
const tfMemoryDOM = {
  tfnumBytes : document.getElementById('tfMemNumBytes'),
  tfnumBytesInGPU : document.getElementById('tfMemNumBytesInGPU'),
  tfnumBytesInGPUAllocated : document.getElementById('tfMemNumBytesInGPUAllocated'),
  tfnumBytesInGPUFree : document.getElementById('tfMemNumBytesInGPUFree'),
  tfnumDataBuffers : document.getElementById('tfMemNumDataBuffers'),
  tfnumTensors : document.getElementById('tfMemNumTensors'),
  tfunreliable : document.getElementById('tfMemUnreliable'),
}

// CHECKBOXES
const greyscaleBox = document.getElementById('greyscale_checkbox');
const downUpscaleBox = document.getElementById('downupscale_checkbox');
const brightnessBox = document.getElementById('brightness_checkbox');

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
  if(model){
    status('Clearing previous model...');
    model.dispose();
  }

  status('Loading model...');

  const startTime = performance.now();
  try{
    if(modelID === 'User Upload'){
      const load = tf.io.browserFiles([userModel.json, ...userModel.weights]);
      model = await tf.loadGraphModel(load, { strict: true });
    } else {
      model = await tf.loadGraphModel(MODELS[modelID], {strict: true});
    }
  } catch (err){
    status("Error loading model!");
    errors(err);
    console.log(err);
    return;
  }

  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  const totalTime = performance.now() - startTime;
  model_upload_time_text.innerText = totalTime;

  status('Model Loaded Successfully.');
  MODEL_LOADED = true;

  const tfMemoryOutput = tf.memory();

  for(const item in tfMemoryOutput){
    const selector = 'tf' + item;
    tfMemoryDOM[selector].innerText = tfMemoryOutput[item];
  }
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
      // This preprocesses the image and returns the model prediction
      // as a tf.Tensor.

      const img = tf.browser.fromPixels(imgElement).toFloat();

      const offset = tf.scalar(127.5);
      const normalized = img.sub(offset).div(offset);

      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      startTime2 = performance.now();
      return model.predict(batched);
    });

    // Furthur processing of the model prediction so that we can display in
    // a HTML canvas.
    const output = await postProcessTF(logits);

    const endTime = performance.now();
    const totalTime = endTime - startTime;
    const preprocessTime = endTime - startTime2;
    updateInferenceTime(totalTime);
    image_preprocess_time_text.innerText = preprocessTime;
    status('Finished Inferencing.');


    tf.browser.toPixels(output, outputCanvas);
  }
}

async function postProcessTF(logits){
  return tf.tidy(() => {
    const scale = tf.scalar(0.5);
    const squeezed = logits.squeeze().mul(scale).add(scale);
    const resized = tf.image.resizeBilinear(squeezed, [IMAGE_SIZE, IMAGE_SIZE]);
    const output = resized.mul(scale).add(scale);

    return output;
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
      vid.onload = () => processVideo(vid);
    };
    reader.readAsDataURL(f);
  }
});

const jsonFileElement = document.getElementById('json-file');
jsonFileElement.addEventListener('change', evt => {
  let files = evt.target.files;
  if(files.length > 1) {
    error('There should only be one JSON file.');
    return;
  }
  for(let i=0, f; f=files[i]; i++){
    if(!f.type === 'application/json'){
      error('Filetype should be JSON!');
      continue;
    }
  }

  userModel.json = files[0];

  parseJSON(userModel.json);

  modelSelect.value = 'User Upload';
  status('Successfully loaded model JSON.');
});

function parseJSON(blob){
  const jsonDOM = {
    convertedBy : document.getElementById('convertedBy'),
    format : document.getElementById('format'),
    generatedBy : document.getElementById('generatedBy'),
    inputTensorname : document.getElementById('inputTensorName'),
    inputTensordtype : document.getElementById('inputTensorType'),
    inputTensortensorShape : document.getElementById('inputTensorShape'),
    outputTensorname : document.getElementById('outputTensorName'),
    outputTensordtype : document.getElementById('outputTensorType'),
    outputTensortensorShape : document.getElementById('outputTensorShape'),
  }


  const jsonReader = new FileReader();
  jsonReader.onload = e => {
    const json = JSON.parse(e.target.result);

    console.log(json);

    const modelTopology = json['modelTopology'];

    for(const item in json){
      if(jsonDOM.hasOwnProperty(item)){
        jsonDOM[item].innerText = json[item];
      }
    }

    const inputs = json.userDefinedMetadata.signature.inputs;
    
    for(const item in inputs){
      for(const input in inputs[item]){
        if(input === 'tensorShape'){
          let st = ""
          for(const d in inputs[item][input]['dim']){
            st = st.concat(inputs[item][input]['dim'][d]['size'], ', ');
          }
          jsonDOM['inputTensortensorShape'].innerText = st;
        } else {
          jsonDOM['inputTensor'+input].innerText = inputs[item][input];
        }
      }
    }

    const outputs = json.userDefinedMetadata.signature.outputs;
    for(const item in outputs){
      for(const output in outputs[item]){
        if(output === 'tensorShape'){
          let st = ""
          for(const d in outputs[item][output]['dim']){
            st = st.concat(outputs[item][output]['dim'][d]['size'], ', ');
          }
          jsonDOM['outputTensortensorShape'].innerText = st;
        } else {
          jsonDOM['outputTensor'+output].innerText = outputs[item][output];
        }
      }
      
    }
  };
  jsonReader.readAsText(blob);
}



const weightsFilesElement = document.getElementById('weights-files');
weightsFilesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  for(let i=0, f; f=files[i]; i++){
    if(!f.type === 'application/octet-stream'){
      error('Wrong Weights filetype!');
      continue;
    }
  }
  userModel.weights = files;
  modelSelect.value = 'User Upload';
  status('Successfully loaded model weights.');
});

function processVideo(videoElement){
  let cap = new cv.VideoCapture(videoElement);
  let frame = new cv.Mat(videoElement.height, videoElement.width, cv.CV_8UC4); // CORRECT

  let dst = new cv.Mat();

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

      preprocessImageCV2(frame, dst, [videoElement.width, videoElement.height], 256, downUpscaleBox.checked, greyscaleBox.checked, brightnessBox.checked);
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

function processImage(imgElement, outputCanvas){
  let img = cv.imread(imgElement);
  let dst = new cv.Mat();

  preprocessImageCV2(img, dst, [imgElement.width, imgElement.height], 256, downUpscaleBox.checked, greyscaleBox.checked, brightnessBox.checked);
  cv.imshow(outputCanvas, dst);

  img.delete();
  dst.delete();
}

modelBtn.addEventListener('click', e => {
  loadModel(modelSelect.options[modelSelect.selectedIndex].value)
});

vid.onplay = () => {
  videoPlaying = true;
  processVideo(vid);
}

vid.onended = () => {
  videoPlaying = false;
}
vid.onpause = () => {
  videoPlaying = false;
}

imgBtn.addEventListener('click', e => {
  if (img.complete && img.naturalHeight !== 0) {
    processImage(img, outputCV2Image);
    predict(outputCV2Image, outputImage);
  } else {
    img.onload = () => {
      processImage(img, outputCV2Image);
      predict(outputCV2Image, outputImage);
    }
  }
});
vidBtn.addEventListener('click', e => {
  vid.play();
  videoPlaying = true;
  processVideo(vid)
});

init();
