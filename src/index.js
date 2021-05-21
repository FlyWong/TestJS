/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

 import '@tensorflow/tfjs-backend-webgl';

 import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
 

 import * as posenet from '@tensorflow-models/posenet';

 import * as posedetection from '@tensorflow-models/pose-detection';
 

 import * as tf from '@tensorflow/tfjs-core';
 
 import {Camera} from './camera';
 
 import {STATE} from './params';

 import {setBackendAndEnvFlags} from './util';
 
 tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
      tfjsWasm.version_wasm}/dist/`);
 

 let detector, camera, stats;
 let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

async function createDetector() {
  /*

switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.MediapipeBlazepose:
      return posedetection.createDetector(STATE.model, {quantBytes: 4});
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
          posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
          posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, {modelType});
  }

  */

  
  STATE.model = posedetection.SupportedModels.MediapipeBlazepose;
  return posedetection.createDetector(STATE.model, {quantBytes: 4});
  

  /*
  return await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 640, height: 480 },
    multiplier: 0.75
  });
  */
  
}

async function renderResult() {
 
  /*
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve();
      };
    });

  */

  

  // FPS only counts the time it takes to finish estimatePoses.
  //beginEstimatePosesStats();

  const poses = await detector.estimatePoses(
      camera.video,
      {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});

  //endEstimatePosesStats();

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (poses.length > 0) {
    camera.drawResults(poses);
  }
}

async function renderPrediction() {
 

 
    await renderResult();
  

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  await tf.setBackend(STATE.backend);

  // Gui content will change depending on which model is in the query string.
  /*

    const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }
  
  */


  



  camera = await Camera.setupCamera(STATE.camera);

  await tf.ready();

  detector = await createDetector();

  renderPrediction();


};

app();




 

 

 

 
