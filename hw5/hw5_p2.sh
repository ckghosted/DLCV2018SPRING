wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1T9c4bo7Y2jlIFwSGFGPmBTyczcupOixC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1T9c4bo7Y2jlIFwSGFGPmBTyczcupOixC" -O model.zip && rm -rf /tmp/cookies.txt
unzip model.zip -d cnn_lstm
python3 hw5_p2.py --video_path "$1" --gen_from cnn_lstm/models_cnn_lstm --label_file "$2" --output_folder "$3"
