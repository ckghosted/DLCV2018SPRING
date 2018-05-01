wget -O model.zip "https://www.dropbox.com/s/gzncos15uoyth5s/models_VGG16_FCN32s.zip?dl=0"
unzip model.zip -d VGG16_FCN32s
python3 inference.py --testing_images "$1" --output_images "$2" --fcn_stride 32 --gen_from VGG16_FCN32s/models
