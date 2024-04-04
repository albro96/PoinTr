import wandb
import random
import numpy as np

# Initialize a new run
wandb.init(project="visualize-predictions", name="point_clouds")

# Generate a cloud of points
points = np.random.uniform(size=(250, 3))

# Log points and boxes in W&B
wandb.log({"test-pcd": wandb.Object3D({"type": "lidar/beta","points": points,})})

wandb.finish()

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 10,
#     }
# )

# # define a metric we are interested in the minimum of
# wandb.define_metric("loss", summary="min")
# # define a metric we are interested in the maximum of
# wandb.define_metric("acc", summary="max")
# for i in range(10):
#     log_dict = {
#         "loss": random.uniform(0, 1 / (i + 1)),
#         "acc": random.uniform(1 / (i + 1), 1),
#     }
#     wandb.log(log_dict)


# # simulate training
# epochs = 10
# ctr = 2
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
#     loss = 2 ** -epoch + random.random() / epoch + offset
#     ctr += 1
#     # log metrics to wandb
#     wandb.log({"acc": acc, "loss": loss})
#     wandb.log({"ctr": ctr})

# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()