{ pkgs }:
let
  caffeModel = pkgs.fetchurl {
    url = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel";
    hash = "sha256-b3EB46IYNzinEloMUCG6gqH+tCKMXKCSTZkbba9vb60=";
  };

  caffeProto = pkgs.fetchurl {
    url = "https://raw.githubusercontent.com/opencv/opencv_extra/20d18acad1bcb312045ea64a239ebe68c8728b88/testdata/dnn/bvlc_googlenet.prototxt";
    hash = "sha256-PPVXbkPq9w/di1mdDavEglwxjfSf+UMNbEjDo2aNlHY=";
  };

  tfBundle = pkgs.fetchzip {
    url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";
    hash = "sha256-Fs7sZVj/uJBk8sHDCzql1odL/Zq5fm9+NfOFw/hBBmA=";
    stripRoot = false;
  };

  onnxModel = pkgs.fetchurl {
    url = "https://github.com/onnx/models/raw/4eff8f9b9189672de28d087684e7085ad977747c/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx";
    hash = "sha256-vZedmdENms+ExaZyUryuSZK/Thr9qwRssOo3Dm8B7jg=";
  };

  ssdModel = pkgs.fetchurl {
    url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel";
    hash = "sha256-KlahGlekopWVawZgtKPXa73KIgbElhzqjv59lcfLLy0=";
  };

  ssdProto = pkgs.fetchurl {
    url = "https://raw.githubusercontent.com/opencv/opencv/4.6.0/samples/dnn/face_detector/deploy.prototxt";
    hash = "sha256-3NZh3Ej8neCjQdsfZmohZOpjpnJlx/d5vBLWs/L6Z+k=";
  };

in pkgs.linkFarm "zigcv-models" [
  { name = "bvlc_googlenet.caffemodel"; path = caffeModel; }
  { name = "bvlc_googlenet.prototxt"; path = caffeProto; }
  { name = "tensorflow_inception_graph.pb"; path = tfBundle + "/tensorflow_inception_graph.pb"; }
  { name = "imagenet_comp_graph_label_strings.txt"; path = tfBundle + "/imagenet_comp_graph_label_strings.txt"; }
  { name = "googlenet-9.onnx"; path = onnxModel; }
  { name = "res10_300x300_ssd_iter_140000.caffemodel"; path = ssdModel; }
  { name = "deploy.prototxt"; path = ssdProto; }
]
