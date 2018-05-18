wget -O model.zip "https://www.dropbox.com/s/v7rqjmnzw0ec198/models_vae.zip?dl=0"
unzip model.zip -d VAE
wget -O model.zip "https://www.dropbox.com/s/ivgl66xbmexhsox/models_gan.zip?dl=0"
unzip model.zip -d GAN
wget -O model.zip "https://www.dropbox.com/s/8f8khzewn2dwwd6/models_acgan.zip?dl=0"
unzip model.zip -d ACGAN
wget -O model.zip "https://www.dropbox.com/s/2lm429s8o01j2ml/models_infogan.zip?dl=0"
unzip model.zip -d INFOGAN
python3 inference_vae.py --testing_images "$1/test" --gen_from VAE/models_vae --output_images "$2"
python3 inference_gan.py --gen_from GAN/models_gan --output_images "$2"
python3 inference_acgan.py --gen_from ACGAN/models_acgan --output_images "$2"
python3 inference_infogan.py --gen_from INFOGAN/models_infogan --output_images "$2"
