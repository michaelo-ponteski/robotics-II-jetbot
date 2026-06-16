# JetBot End-to-End Autonomous Driving

## Team members
1. Michał Pokładowski 160278
2. Marek Olejniczak 160303
3. Tomasz Preś 160305
4. Karol Okoń 160323

Robotics II project report.


The goal of this project was to make a JetBot drive autonomously around a fixed track. We used an end-to-end (behavioral cloning) approach: a single neural network takes one camera image and directly predicts the motor commands. There is no separate lane detection or path planning, the network learns everything from the data we collected by driving the robot ourselves.

## Approach

The whole pipeline is a loop of three steps that we repeated several times:

1. **Data collection** (`jazda3.ipynb`, runs on the JetBot). We drive the robot manually with a gamepad and record the camera frames together with the controls we send to the motors. Each frame is saved as a JPG and the controls are written to a CSV file. We record at about 10 Hz.
2. **Training** (`training.ipynb`, runs on a PC with a GPU). We take all the recorded sessions, train the CNN, and export it to ONNX format.
3. **Autonomous driving** (`jazda_model.ipynb`, runs on the JetBot). The exported model is loaded with ONNX Runtime and drives the robot in real time, also at 10 Hz.

The network output is two continuous values: `forward` (speed) and `left` (steering), both in the range `<-1, 1>`. We treat this as a regression problem.

One important detail is the **latency compensation**. Because inference and motor reaction take some time, predicting the control for the current frame is actually a bit too late. So during training we shift the targets: for the image at frame `i` the target is the control from frame `i + k` (we used `k = 2`). This way the model learns to output the command that will be needed slightly in the future. The same constant has to be used in the inference notebook, otherwise the behavior does not match.

We also added a "turbo" mechanism during data collection. The normal forward speed is limited (scale 0.5), but holding a trigger button blends the speed up to full. The scaled value is what we actually save as the label, so the model learns the real speed we used. During autonomous driving there is a slider that caps the forward speed, which let us test the model safely at low speed first and then speed it up.

## Dataset

We collected the data ourselves by driving the robot around the track in both directions and in different lighting. After cleaning, the dataset contains:

- **9 sessions**, **7021 frames** in total
- Each sample is a 224×224 RGB image and a label `(forward, left)`

The dataset is heavily unbalanced, most of the track is straight, so most labels have `left ≈ 0`. The `forward` value is also concentrated around 0.5 (normal cruising speed). Because of this imbalance we had to be careful in training (see below).

Before training, each session is cleaned in memory: duplicate indices are removed, rows with missing or out-of-range values are dropped, and rows whose image file does not exist are removed. The original CSV files are not modified.

For the train/validation split we did **not** use a simple random per-frame split. Neighboring frames look almost identical, so a random split would leak the validation data into training. Instead we cut each session into contiguous blocks of 20 frames and then randomly assign whole blocks to train or validation (80/20). This keeps all parts of the track in both sets but avoids the leakage. This gave us about 5700 training and 1300 validation frames.

To make the model more robust we used several **augmentations** during training (only on the training set):

- brightness, contrast and gamma changes (different lighting conditions)
- a small random crop (simulates the robot being a few percent off-center)
- horizontal flip, which also flips the sign of the `left` label (this also helps because turns are rare)

We did not use saturation or hue augmentation on purpose, because the track has fixed colors and changing them would not be realistic.

## Model architecture

The model is a convolutional neural network inspired by NVIDIA's PilotNet, adapted for 224×224 input. It has about **540k parameters**.

- **Feature extractor:** 5 convolutional layers, each followed by batch normalization and ReLU. The spatial size is reduced step by step: 224 → 112 → 56 → 28 → 14 → 7. The channels grow 3 → 24 → 36 → 48 → 64 → 64.
- **Regression head:** flatten (64×7×7 = 3136 features) → fully connected 128 → 64 → 2, with dropout (0.2) for regularization.
- **Output:** a `tanh` activation keeps both outputs in `<-1, 1>`.

Training setup:

- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4
- Scheduler: cosine annealing over 40 epochs
- Batch size: 128, gradient clipping at norm 1.0
- **Weighted MSE loss:** because straight driving dominates, a normal loss would basically ignore the steering. So we weight the `left` component much more than `forward` (0.8 vs 0.2). This forces the model to actually learn the turns.

We save the checkpoint with the best validation loss and export it to ONNX (OPSET 11, fixed batch `(1, 3, 224, 224)`, float32, RGB in `[0, 1]`).

## Error analysis

The best validation results were around epoch 29:

- **Validation loss (weighted MSE): ~0.043**
- **Forward MAE: ~0.073**
- **Left (steering) MAE: ~0.17**

A few observations:

- The `forward` value is predicted very accurately (MAE under 0.08). This is expected because most of the time the speed is roughly constant, so it is an easy target.
- The `left` value is harder. The steering MAE is around 0.17, which is significant compared to the turn values (turns are about ±0.6). This is the main source of error and the main thing limiting the driving quality.
- The reason is the **class imbalance**: there are far fewer turning frames than straight frames. Even with the weighted loss, the model is conservative and tends to under-steer in sharp turns. We could see this in the sanity-check visualization, where straight and gentle turns were predicted well, but the strongest turns sometimes had a too-small predicted value.
- Training and validation loss stayed close to each other, so the model is not strongly overfitting. The augmentations and dropout probably helped here.
- On the real robot, the model drove the track reliably at the normal speed cap. At higher speeds the under-steering became more visible, because the robot has less time to correct in a turn.
<img width="1429" height="1033" alt="image" src="https://github.com/user-attachments/assets/8898c343-1073-4390-ad73-e7247469b20b" />


## Conclusions

The end-to-end approach worked: with a fairly small CNN and around 7000 collected frames, the JetBot was able to drive the track autonomously. The most important factors were:

- **Latency compensation** (predicting `k` frames ahead) — without it the steering felt delayed.
- **Block-based train/val split** — a naive random split gave unrealistically good validation numbers because of data leakage.
- **Weighted loss and flip augmentation** — needed to deal with the imbalance between straight and turning frames.

The main weakness is steering in sharp turns, caused by the dataset imbalance. If we continued the project, the next steps would be:

- collect more data specifically in the difficult turns,
- try oversampling or stronger weighting of turning frames,
- possibly add a short history of frames so the model has some temporal context instead of a single image.

## Repository contents

| File | Description |
| --- | --- |
| `jazda3.ipynb` | Data collection on the JetBot (gamepad + recording) |
| `training.ipynb` | Offline training and ONNX export (run on a GPU PC) |
| `jazda_model.ipynb` | Autonomous driving on the JetBot using the ONNX model |
| `jetbot_model.onnx` | Trained model, ready for inference |

The data-collection and driving notebooks require the JetBot hardware (the `jetbot` and `PUTDriver` libraries), so they only run on the robot. Only the training notebook runs on a normal computer.
