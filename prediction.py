# from ultralytics import YOLO

# #Initialize YOLO with the Model Name
# model = YOLO("best.pt")

# ##Predict Method Takes all the parameters of the Command Line Interface

# model.predict(source='person.jpg', save=True, conf=0.25, save_txt=True)
# # model.export(format="onnx")

from ultralytics import YOLO
import cv2

# Initialize YOLO with the model
model = YOLO("person-animals.pt")

# Open the webcam (source='0' for the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model.predict(source=frame, conf=0.25, save=False, show=False)

    # Extract annotated frame
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels on the frame

    # Display the annotated frame
    cv2.imshow("YOLOv8 - Webcam", annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
