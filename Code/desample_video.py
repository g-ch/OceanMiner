import cv2
 
start_time = 33 # second
end_time = 42 # second
new_fps = 2
videoCapture = cv2.VideoCapture("../test.MP4")
new_path = "../test_desample_33_42.MP4"
if_show_video = 0

#Read fps to give a waiting duration
fps = videoCapture.get(cv2.CAP_PROP_FPS)
#Read size to keep original size
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
retval  =   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

# I420-avi, MJPG-mp4
videoWriter = cv2.VideoWriter(new_path, retval, new_fps, size)
 
#Read
success, frame = videoCapture.read()
fps_counter = 1
time_counter = 0
record_flag = 0

while success :

    if if_show_video == 1:
        cv2.imshow("Video", frame) 
        cv2.waitKey(1000/int(fps)) 

    # Update time counter
    if fps_counter >= int(fps):
        fps_counter = 1
        time_counter = time_counter + 1

    #Update record flag
    if time_counter >= start_time:
        record_flag = 1
    
    if time_counter >= end_time:
        break

    #Save image
    if fps_counter == int(fps / new_fps) and record_flag == 1:   #Desample to new fps
        videoWriter.write(frame)
        fps_counter = 0

    fps_counter = fps_counter + 1
    success, frame = videoCapture.read()
    
videoWriter.release()
