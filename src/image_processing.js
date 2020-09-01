export function preprocessImageCV2(cv2mat, dst, size, targetSize=256, downUpscale, desaturate){
  let output = cv2mat.clone();

  if(size[0] !== 256 || size[1] !== 256){
    const offset_x = Math.floor((size[0]- targetSize) / 2);
    const offset_y = Math.floor((size[1] - targetSize) / 2);
    const mask = new cv.Rect(offset_x, offset_y, targetSize, targetSize);

    output = output.roi(mask);
  }

  const smallSize = Math.floor(targetSize / 24);
  const dsize = new cv.Size(smallSize, smallSize);
  const usize = new cv.Size(targetSize, targetSize);

  if(downUpscale){
    cv.resize(output, output, dsize, 0, 0, cv.INTER_AREA); // Downscale
    cv.resize(output, output, usize, 0, 0, cv.INTER_CUBIC); // Upscale
  }
  if(desaturate){
    cv.threshold(output, output, 150, 200, cv.THRESH_TRUNC); // Desaturate
    //cv.cvtColor(dst, dst, cv.COLOR_RGBA2GRAY, 3); // Grayscale
  }
  //cv.imshow(outputCV2Video, dst);
  output.copyTo(dst);
  output.delete();
}
