# Handball AI: Player Detection, Tracking, and Identification

**Detect, track, and identify handball players in videos using computer vision.**

This project is **adapted from [Roboflow's Basketball Player Detection Notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb)** and repurposed for handball. It demonstrates a complete pipeline for detecting, tracking, and identifying handball players in videos using **RF-DETR** for object detection, **SAM2** for real-time player tracking, **SigLIP** for team classification, and **SmolVLM2** for jersey number recognition. The pipeline also maps player positions to court coordinates for advanced analytics.

It is in active development. Current status is shown below.

I have already fine-tuned RF-DETR on a handball-player detection dataset. The detection of players already works quite well.
However I still need to adapt some features, which will include fine-tuning other models to handball scenarios.

---

## Current Features

- **Player Detection**: Detect players, referees, and the ball using a fine-tuned RF-DETR model.
- **Player Tracking**: Track players across frames with stable IDs and masks using SAM2.

## Future Features

- **Team Classification**: Automatically cluster players into teams using SigLIP embeddings and K-means.
- **Jersey Number Recognition**: Recognize and validate player numbers using SmolVLM2 OCR.
- **Court Mapping**: Map player positions to real-world court coordinates.
- **Visualization**: Overlay player names, numbers, team colors, and movement paths on video.

## Project Status

- **RF-DETR** has been fine-tuned on a handball-player detection dataset.
- SAM2 player tracking works using the prompt from the RF-DETR detection.
- Additional features are being adapted for handball scenarios.


---

## References and Acknowledgements

- **[Roboflow Basketball Player Detection Notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb)**: This project is based on Roboflow's basketball pipeline, adapted for handball.
- [Roboflow Universe](https://universe.roboflow.com/) for pre-trained models, fine-tuning models and tools useful for dataset creation.
- [SAM2 Real-Time](https://github.com/Gy920/segment-anything-2-real-time) for real-time segmentation.
- [Roboflow Sports](https://github.com/roboflow/sports) for team classification and court mapping tools.

---

## 🤝 Contributing

Found a bug or have an idea for improvement? Open an issue or submit a pull request!
