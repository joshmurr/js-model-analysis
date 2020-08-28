export function getExtension(filename){
  let parts = filename.split('.');
  return parts[parts.length - 1];
}

export function isVideo(filename) {
  var ext = getExtension(filename);
  switch (ext.toLowerCase()) {
    case 'm4v':
    case 'avi':
    case 'mpg':
    case 'mp4':
      // etc
      return true;
  }
  return false;
}
