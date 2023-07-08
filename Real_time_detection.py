import cv2
from pyzbar import pyzbar
from pathlib import Path

def readBarcodes(videoFrame):
    # make grayscale
    grayFrame = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)

    # find barcodes
    detectedBarcodes = pyzbar.decode(grayFrame)

    for barcode in detectedBarcodes:
        # find conners
        (x, y, w, h) = barcode.rect

        # Draw green rectangle for barcode
        cv2.rectangle(videoFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # get info
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # show info
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(videoFrame, text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

    return videoFrame

# select default camera
video = cv2.VideoCapture(0)

# video size
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create the codec and VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_path = Path("outputVideos")
if not output_path.exists():
    # create output directory
    output_path.mkdir(parents=True, exist_ok=True)

output_path = 'outputVideos/outV1.avi'
output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

while True:
    ret, frame = video.read()

    # Stop if there is no frame
    if not ret:
        break

    output_frame = readBarcodes(frame)
    output_video.write(output_frame)

    cv2.imshow('Barcode Detection', output_frame)

    # if e is pressed exit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# exit
output_video.release()
video.release()
cv2.destroyAllWindows()