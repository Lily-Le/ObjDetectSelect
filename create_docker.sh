docker build -t object-detect .
docker run --gpus all -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix docker-fire-detect
python detect.py --source dataset/fire.png --weights ./best.pt