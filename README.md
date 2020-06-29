# eccv2020
* copy `/u/lchen63/Project/face_tracking_detection/eccv2020/basics` under this folder. We need some dependences in basics forder.
* The video frame cropper for talking head is in `./data/face_tracker.py` line 72  `_crop_video(video, pid = 0)` function.



For data preprocess:

data/single_video_preprocess.py 

	use landamrk_extractor to extract landmarks.
	use computeRT to compute the RT betwen the canonical landmark and target landmark

	use get_front_video to find the front face to represent the video.

	modify get_3d and copy it to ~/github/PRNet and run under py36 environemnt.

	go to pytorch1.1python3 conda environment,
	use face_tool find_camera to generate rendered 3D video.
