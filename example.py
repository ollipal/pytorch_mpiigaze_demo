import cv2
from saurongaze import SauronGaze


if __name__ == "__main__" :
    cap = cv2.VideoCapture(0)
    sgaze = SauronGaze()
    cv2.namedWindow("saurongaze", cv2.WINDOW_NORMAL)

    while(True):
        # read frame form the camera
        ret, frame = cap.read()
        if not ret:
            print("no frame returned")
            break

        # get and log head + gaze information
        head, gaze = sgaze.refresh(frame)
        print(f"Head:\n{head}\n")
        print(f"Gaze:\n{gaze}\n")

        # get and show drawn frame
        drawn_frame = sgaze.get_drawn_frame(
            draw_head=True,
            draw_gaze=True
        )        
        cv2.imshow("saurongaze", drawn_frame)

        # end program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' was pressed")
            break