wget -O model.zip "https://www.dropbox.com/s/1mxqmi506y3hfhe/models_dilated.zip?dl=0"
unzip model.zip -d VGG16_dilated
python3 inference_dilated.py --testing_images "$1" --output_images "$2" --gen_from VGG16_dilated/models
