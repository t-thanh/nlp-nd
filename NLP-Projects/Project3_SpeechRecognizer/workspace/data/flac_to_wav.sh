# Convert all .flac files within this folder to .wav files

find . -iname "*.flac" | wc

for flacfile in `find . -iname "*.flac"`
do
    ffmpeg -y -f flac -i $flacfile -ab 64k -ac 1 -ar 16000 -f wav "${flacfile%.*}.wav"
done

find . -type f -iname \*.flac -delete
