import cv2
from saurongaze import SauronGaze

if __name__ == "__main__" :
    cap = cv2.VideoCapture(0)
    sgaze = SauronGaze()
    cv2.namedWindow("saurongaze", cv2.WINDOW_NORMAL)

    while(True) :
        ret, frame = cap.read()
        if not ret:
            print("no frame returned")
            break

        sgaze.refresh(frame)
        drawn_frame = sgaze.get_frame()

        cv2.imshow("saurongaze", drawn_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' was pressed")
            break