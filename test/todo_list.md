## Todo List to Improve the Performance

### Calculated Image

#### The most basic example
Split training and validation data by simply using 'random_split'.
**result: wandb -> Sperated-Categories-Angles -> Test-first-time**
Class_acc: 88.636, angle_loss: 0.1187

#### Adjust the dataset by splitting the category with only two samples into training and validation data separately
**script: test/test_second_time.py**
**result: wandb -> Sperated-Categories-Angles -> Test-second-time**
Class_acc: 83.721, angle_loss: 0.1025

#### Applying `Focal Loss` for unbalanced dataset.
**script: test/test_third_time.py**
**result: wandb -> Sperated-Categories-Angles -> Test-third-time-gamma-3.5-no-class-weights**
Class_acc: 90.698, angle_loss: 0.1064

#### Applying the transformations of rotating to the dataset.


#### ToDos
- [x] Basic implementation
- [x] Adjust the dataset by splitting the category with only two samples into training and validation data separately.
- [x] Applying `Focal Loss` for unbalanced dataset.
- [ ] Resample the dataset by rotating the ref images.
- [ ] Applying the transformations of rotating to the dataset.