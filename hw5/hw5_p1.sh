wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v8yrebWBY3xvVFrnuUBurUIOKhgYHcY7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v8yrebWBY3xvVFrnuUBurUIOKhgYHcY7" -O model.zip && rm -rf /tmp/cookies.txt
unzip model.zip -d cnn_dnn
python3 hw5_p1.py --video_path "$1" --gen_from cnn_dnn/models_cnn_dnn --label_file "$2" --output_folder "$3"
