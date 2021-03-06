wget -O model.zip "https://www.dropbox.com/s/ynnjuj2mvku8605/models_VGG16_FCN8s.zip?dl=0"
unzip model.zip -d VGG16_FCN8s
python3 inference.py --testing_images "$1" --output_images "$2" --fcn_stride 8 --gen_from VGG16_FCN8s/models
