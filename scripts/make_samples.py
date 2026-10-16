from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


_SCENES = [
    ("dog_grass", "a dog running through grass"),
    ("person_ball", "a person holding a red ball"),
    ("two_cats", "two cats looking at the sky"),
    ("train_station", "a red train at a station"),
    ("puppy_water", "a small puppy jumping in water"),
    ("tree_dog", "a dog sitting near a tree"),
    ("beach_family", "a family on a beach"),
    ("city_street", "a busy city street at sunset"),
    ("bird_tree", "a bird perched on a branch"),
    ("surfing", "a surfer riding a wave"),
    ("mountain_hike", "people hiking in the mountains"),
    ("market", "a market stall with colorful goods"),
]


def _make_scene(size, scene_name):
    image = Image.new("RGB", size, color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    if scene_name == "dog_grass":
        draw.rectangle((0, 150, size[0], size[1]), fill=(120, 180, 100))
        draw.ellipse((80, 70, 170, 140), fill=(150, 110, 75))
        draw.rectangle((110, 140, 150, 190), fill=(160, 120, 80))
        draw.rectangle((75, 190, 110, 220), fill=(160, 120, 80))
        draw.rectangle((145, 190, 180, 220), fill=(160, 120, 80))
    elif scene_name == "person_ball":
        draw.rectangle((0, 170, size[0], size[1]), fill=(120, 160, 100))
        draw.ellipse((85, 60, 150, 120), fill=(220, 185, 142))
        draw.line((120, 120, 120, 190), fill=(70, 70, 70), width=6)
        draw.line((120, 150, 170, 200), fill=(70, 70, 70), width=6)
        draw.line((120, 150, 80, 200), fill=(70, 70, 70), width=6)
        draw.ellipse((175, 135, 205, 165), fill=(220, 40, 40))
    elif scene_name == "two_cats":
        draw.rectangle((0, 180, size[0], size[1]), fill=(80, 150, 200))
        draw.ellipse((70, 70, 120, 120), fill=(150, 140, 130))
        draw.ellipse((150, 70, 200, 120), fill=(150, 140, 130))
        draw.polygon([(110, 120), (135, 150), (90, 150)], fill=(150, 140, 130))
    elif scene_name == "train_station":
        draw.rectangle((0, 180, size[0], size[1]), fill=(160, 170, 150))
        draw.rectangle((40, 100, 180, 170), fill=(220, 70, 70))
        draw.rectangle((60, 120, 160, 160), fill=(250, 250, 250))
        draw.rectangle((200, 110, 250, 170), fill=(90, 90, 220))
    elif scene_name == "puppy_water":
        draw.rectangle((0, 170, size[0], size[1]), fill=(70, 140, 200))
        draw.ellipse((90, 80, 170, 150), fill=(200, 150, 100))
        draw.polygon([(115, 150), (145, 180), (85, 180)], fill=(200, 150, 100))
    elif scene_name == "tree_dog":
        draw.rectangle((0, 170, size[0], size[1]), fill=(120, 170, 100))
        draw.rectangle((150, 40, 170, 170), fill=(120, 80, 40))
        draw.ellipse((130, 20, 190, 80), fill=(70, 170, 100))
        draw.ellipse((70, 80, 130, 150), fill=(160, 120, 80))
    elif scene_name == "beach_family":
        draw.rectangle((0, 150, size[0], size[1]), fill=(110, 200, 220))
        draw.rectangle((0, 150, size[0], size[1]), fill=(100, 190, 200))
        draw.circle((80, 105), 30, fill=(240, 200, 150))
        draw.circle((120, 100), 28, fill=(240, 200, 150))
        draw.rectangle((90, 130, 165, 170), fill=(200, 70, 60))
    elif scene_name == "city_street":
        draw.rectangle((0, 140, size[0], size[1]), fill=(180, 160, 150))
        draw.rectangle((40, 40, 80, 140), fill=(90, 90, 100))
        draw.rectangle((100, 50, 150, 140), fill=(100, 110, 140))
        draw.rectangle((180, 70, 220, 140), fill=(130, 90, 100))
        draw.rectangle((0, 0, size[0], 30), fill=(255, 180, 100))
    elif scene_name == "bird_tree":
        draw.rectangle((0, 180, size[0], size[1]), fill=(120, 180, 110))
        draw.rectangle((150, 40, 170, 180), fill=(120, 80, 40))
        draw.ellipse((130, 20, 200, 90), fill=(60, 170, 110))
        draw.ellipse((60, 90, 100, 120), fill=(220, 200, 120))
    elif scene_name == "surfing":
        draw.rectangle((0, 140, size[0], size[1]), fill=(110, 180, 220))
        draw.arc((30, 40, 200, 180), 200, 360, fill=(240, 240, 255), width=8)
        draw.line((150, 110, 200, 170), fill=(60, 80, 160), width=5)
    elif scene_name == "mountain_hike":
        draw.rectangle((0, 170, size[0], size[1]), fill=(145, 170, 120))
        draw.polygon([(40, 180), (100, 80), (160, 180)], fill=(110, 110, 120))
        draw.polygon([(120, 180), (190, 70), (250, 180)], fill=(130, 130, 140))
        draw.ellipse((190, 60, 230, 95), fill=(220, 200, 170))
    elif scene_name == "market":
        draw.rectangle((0, 160, size[0], size[1]), fill=(220, 180, 130))
        for x in [40, 100, 160, 220]:
            draw.rectangle((x, 70, x + 30, 170), fill=(220, 120, 80))
        draw.rectangle((0, 0, 224, 25), fill=(210, 90, 70))
    return image


def main():
    samples_dir = Path("artifacts/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, (scene_name, caption) in enumerate(_SCENES):
        image = _make_scene((224, 224), scene_name)
        path = samples_dir / f"{scene_name}.png"
        image.save(path)
        manifest.append({"image": path.name, "scene": scene_name, "caption": caption})
    (samples_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(manifest)} sample images under {samples_dir}")


if __name__ == "__main__":
    main()
