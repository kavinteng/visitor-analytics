import torch,cv2,dlib,time,datetime,os
import numpy as np
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from threading import Thread

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def create_folder():
    base_dir = os.path.dirname(os.path.abspath('__file__'))
    backup_vdo = os.path.join(base_dir, "backup_video")
    backup_img = os.path.join(base_dir, "backup_img")
    date_img = os.path.join(backup_img, "{}".format(datetime.date.today()))
    date_vdo = os.path.join(backup_vdo, "{}".format(datetime.date.today()))

    if os.path.isdir(backup_img) == False:
        os.mkdir(backup_img)
    if os.path.isdir(date_img) == False:
        os.mkdir(date_img)
    if os.path.isdir(backup_vdo) == False:
        os.mkdir(backup_vdo)
    if os.path.isdir(date_vdo) == False:
        os.mkdir(date_vdo)

def main(rtsp,device,save_video = False,cap_person_roi = False):
    cap = cv2.VideoCapture(rtsp)
    st = None
    record = 0

    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=8, maxDistance=90)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalout = 0
    totalin = 0
    x = []
    empty = []
    empty1 = []

    while True:
        create_folder()
        date = datetime.date.today()
        ret,frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame,(640,360))
        frame_record = frame.copy()
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = []

        if st == None:
            st = time.time()
        et = time.time()

        if et - st > 0.1:
            # if totalFrames % 2 == 0:
            trackers = []

            roi = frame[0:H,int((W/2)-100):int((W/2)+100)]
            (H_roi, W_roi) = roi.shape[:2]
            results = model(roi, size=480)

            out2 = results.pandas().xyxy[0]

            if len(out2) != 0:
                rects = []
                for i in range(len(out2)):
                    xmin = int(out2.iat[i, 0])
                    ymin = int(out2.iat[i, 1])
                    xmax = int(out2.iat[i, 2])
                    ymax = int(out2.iat[i, 3])
                    obj_name = out2.iat[i, 6]

                    if obj_name != 'person':
                        continue
                    if obj_name == 'person' or obj_name == '0':
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(xmin, ymin, xmax, ymax)
                        tracker.start_track(rgb, rect)

                        trackers.append(tracker)

                        if cap_person_roi == True:
                            person_img = frame_record[ymin:ymax, xmin:xmax]
                            b = datetime.datetime.now().strftime("%T")
                            b = b.replace(':', '-')
                            b = str(b)
                            filename = f'backup_img/{date}/device' + str(device) + f"t{b}.jpg"
                            cv2.imwrite(filename, person_img)

            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

            cv2.line(frame, ((W // 2) + 0, 0), ((W // 2) + 0, H), (0, 0, 0), 3)
            cv2.line(frame, ((W // 2) - 100, 0), ((W // 2) - 100, H), (0, 0, 0), 3)
            cv2.line(frame, ((W // 2) + 100, 0), ((W // 2) + 100, H), (0, 0, 0), 3)

            boundingboxes = np.array(rects)
            boundingboxes = boundingboxes.astype(int)
            rects = non_max_suppression_fast(boundingboxes, 0.3)
            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)

                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    y = [c[0] for c in to.centroids]

                    direction = centroid[0] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        if direction < -20 and ((W_roi // 2) - 100 < centroid[0] < W_roi // 2):
                            totalin += 1
                            print(objectID,direction)
                            to.counted = True

                        elif direction > 20 and ((W_roi // 2) + 100 > centroid[0] > W_roi // 2):
                            totalout += 1
                            print(objectID,direction)
                            to.counted = True

                trackableObjects[objectID] = to

                objectID = objectID + 1
                # cv2.rectangle(roi, (x1 - 5, y1), (x2 - 5, y2), (0, 0, 255), 2)
                text = "ID {}".format(objectID)
                cv2.putText(roi, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(roi, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

            info = [
                ("Exit", totalout),
                ("Enter", totalin),
            ]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if save_video == True:
                if record == 0:
                    a = datetime.datetime.now().strftime("%T")
                    a = a.replace(':', '-')
                    a = str(a)

                    file = f'backup_video/{date}/device' + str(device) + "t{}.mp4".format(a)

                    video_size = (640, 360)
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    rec = cv2.VideoWriter(file, fourcc, 30, video_size)

                    record = 1

                if record == 1:
                    rec.write(frame_record)
            st = time.time()
            cv2.imshow(f'{rtsp}', frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        totalFrames += 1
    cap.release()
    if record == 1:
        rec.release()
    cv2.destroyWindow(f'{rtsp}')

def main_threading(rtsp,device,save_video,cap_person_roi):
    t1 = Thread(target=main, args=(rtsp,device,save_video,cap_person_roi,))
    t1.start()

if __name__ == '__main__':
    print('start load model!!!')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.2
    model.iou = 0.9

    print('load yolov5 successfully!!!')

    main_threading(rtsp='rtsp://test:advice128@110.49.125.237:554/cam/realmonitor?channel=1&subtype=0',
                   device=1,
                   save_video=False,
                   cap_person_roi=False)

    # main_threading(rtsp='rtsp://admin:888888@192.168.7.50:10554/tcp/av0_0',
    #                device=2,
    #                save_video=False,
    #                cap_person_roi=False)