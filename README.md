## JS cGAN Analysis and Benchmarking

A web tool to upload pre-trained GANs and to analyse how well the models perform on the web through simple benchmarking and analysis tools. The overall aim to test different methods of model compression and ways of acheiving real-time interaction on the web.

[Auto-pix2pix](https://github.com/joshmurr/cci-auto-pix2pix) is a version of the [pix2pix](https://arxiv.org/pdf/1611.07004.pdf) model based on [Learning to See](https://arxiv.org/ftp/arxiv/papers/2003/2003.00902.pdf) by Memo Akten et al which set up to quickly protoype generative models for this platform. If you would like to quickly create a dataset from video and train your own model which would be compatible here, see [Auto-pix2pix here](https://github.com/joshmurr/cci-auto-pix2pix).

---

### To Run Locally:

Some sample images and video are included in the repository if a webcam is not available.

```bash
# Clone the Repo
$ git clone https://github.com/joshmurr/js-model-analysis && cd js-model-analysis
$ npm install

# Download Sample Model:
$ mkdir dist/models/greyscale2flowers
$ wget https://transfer.sh/plchj/greyscale2flowers.tar.gz
$ tar -xf greyscale2flowers.tar.gz -C dist/models/greyscale2flowers

# NB: At the time of creating this transfer.sh has a notice on thier website
# suggesting the service will be dropped from 30-09-20. If you're looking at 
# this past that date, I suggest training your own using Auto-pix2pix.

# Run
$ npm run dev
# View at http://localhost:8080/
```

---

A video of the site in use:

[![Site in Use](https://img.youtube.com/vi/JsSXUqzfHrY/0.jpg)](https://www.youtube.com/watch?v=JsSXUqzfHrY)
