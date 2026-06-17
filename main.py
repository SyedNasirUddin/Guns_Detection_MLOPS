import io

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
from torchvision import transforms

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# LOAD CUSTOM TRAINED MODEL
# =====================================================

NUM_CLASSES = 2  # background + gun

model = fasterrcnn_mobilenet_v3_large_fpn(
    weights=None
)

in_features = (
    model.roi_heads.box_predictor
    .cls_score.in_features
)

model.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )
)

model.load_state_dict(
    torch.load(
        "artifacts/models/fasterrcnn.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("CUSTOM GUN DETECTOR LOADED SUCCESSFULLY")

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.ToTensor()
])

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()


def predict_and_draw(image: Image.Image):

    image = image.convert("RGB")

    img_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    output_image = image.copy()

    draw = ImageDraw.Draw(output_image)

    gun_found = False

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        # Gun class = 1
        if label == 1 and score > 0.50:

            gun_found = True

            x_min, y_min, x_max, y_max = box

            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline="red",
                width=4
            )

            draw.text(
                (x_min, y_min),
                f"Gun {score:.2f}",
                fill="red"
            )

    print(
        f"Detections: {len(boxes)} | Gun Found: {gun_found}"
    )

    return output_image


@app.get("/")
def home():

    return {
        "message": "Gun Detection API Running"
    }


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    output_image = predict_and_draw(
        image
    )

    img_byte_arr = io.BytesIO()

    output_image.save(
        img_byte_arr,
        format="PNG"
    )

    img_byte_arr.seek(0)

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png"
    )