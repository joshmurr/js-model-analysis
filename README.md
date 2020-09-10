## JS cGAN Analysis and Benchmarking

A web tool to upload pre-trained GANs and to analyse how well the models perform on the web through simple benchmarking and analysis tools. The overall aim to test different methods of model compression and ways of acheiving real-time interaction on the web.

[Auto-pix2pix][auto-p2p] is a version of the [pix2pix](https://arxiv.org/pdf/1611.07004.pdf) model based on [Learning to See](https://arxiv.org/ftp/arxiv/papers/2003/2003.00902.pdf) by Memo Akten et al which set up to quickly protoype generative models for this platform. If you would like to quickly create a dataset from video and train your own model which would be compatible here, see [Auto-pix2pix here](https://github.com/joshmurr/cci-auto-pix2pix).

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

### Convert a Pretrained Model to Upload

From your Tensorflow Python script save the model in the `.h5` format.

```python
model.save('model.h5')
```

We're only interested in the generator of the _pix2pix_ model so any generative model which matches the input and output dimensions (256x256x[1..3]) should work. The channel dimension is adapted as the model loads so RGB or greyscale will work. However I have only tested it with _pix2pix_ style models trained using [Auto-pix2pix][auto-p2p] so I can't say for sure just yet.

[Follow the instructions to install TensorflowJS_Converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter). From a virtual environment run `tensorflowjs_wizard`, or this command should work:

```bash
tensorflowjs_converter --input_format=keras --output_format=tfjs_graph_model --weight_shard_size_bytes=4194304 {{Path to Model}}.h5 {{Output Path}}
```

The `tensorflowjs_wizard` provides you with some handy options for compression and other things. If using a model from _Auto-pix2pix_ make sure to choose __Graph Layers__ for the output format.

Once completed, in your chosen output folder you should see a `model.json` file and a bunch of binary files. You should now be able to upload them to the `js-model-analyser` via the menu at the top of the page (uploading `model.json` and the binaries seperately) - then press _load model_.

---

A video of the site in use:

<div style="display: flex; justify-items: center;">
[![Site in Use](https://img.youtube.com/vi/JsSXUqzfHrY/0.jpg)](https://www.youtube.com/watch?v=JsSXUqzfHrY)
</div>

<!-- -->

[auto-p2p]: https://github.com/joshmurr/cci-auto-pix2pix
