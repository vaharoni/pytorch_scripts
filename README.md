# pytorch_scripts

The main goal of this framework is to allow re-running training scripts up to an arbitrary point without incurring the cost of training the model from scratch. This is achieved by saving checkpoint files that behave as cache for re-runs. 

## Installation
Remotely:
```bash
pip install git+https://github.com/vaharoni/pytorch_scripts.git
```

Locally, allowing for any edits in this package's code to take effect while developing a project that depends on it:
```bash
git clone https://github.com/vaharoni/pytorch_scripts.git

cd some_dependent_project
pip install -e ../pytorch_scripts
```

Put this in the `.gitignore` file:
```text
runs/
ckpts/
```

## Synopsis
```python
from pytorch_scripts import train

# model_builder() should return an nn.Module object
model = train.Model('model-id', model_builder(), 'Model description')

# Train model
with model.checkpoint(id=1) as cp:
    cp.setup(
        loss_cls=torch.nn.CrossEntropyLoss, 
        optimizer_cls=torch.optim.Adam,
        optimizer_args=dict(lr=5e-4, weight_decay=5e-4),
        epoch_scheduler_cls=torch.optim.lr_scheduler.StepLR, 
        epoch_scheduler_args=dict(step_size=10, gamma=0.2)
    )
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])
    cp.plot_metrics()

# In preparation for the next training cycle, plot loss vs. learning rate using a different optimizer
model.cache(id=1).find_lr(train_dl, optimizer_cls=torch.optim.SGD)

# Train some more with a different optimizer
with model.checkpoint(2) as cp:
    cp.setup(
        optimizer_cls=torch.optim.SGD,
        optimizer_args=dict(lr=0.001),
        clear_epoch_scheduler=True
    )
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])    
    cp.plot_metrics()

# Rollback to the post-training state of checkpoint 1:
# Reinstantiate the training components (Adam optimizer, epoch learning scheduler), load their states, and load the model parameters 
model.load_checkpoint(id=1)

# Calling a new training block with a different checkpoint ID effectively performs a fork.
# Now checkpoints 2 and 3 are the result of different training approaches from checkpoint 1.
# The training process continues exactly where checkpoint 1 ended.
with model.checkpoint(3) as cp:
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=40, metrics=[train.metric_accuracy])
    cp.plot_metrics()

# Measure the model's performance
model.cache(3).evaluate(test_dl, metrics=[train.metric_accuracy])

# Deletes all files asssociated with checkpoint 1
model.delete_checkpoint(1)

# Deletes the model and all its checkpoints
model.delete()
```

Each checkpoint stores in a few files the:
- model parameters
- optimizer state
- epoch scheduler state - a scheduler whose step() function is called per training epoch
- step scheduler state - a scheduler whose step() function is called per training step
- results from various cachable operations

There are two main APIs for checkpoints - the training API and the cache API. 

## The Training API

The training API is invoked using a `with` block:
```python
with model.checkpoint(id=1, description='Adam for 20 + 20 epochs') as cp:
    cp.setup(
        loss_cls=torch.nn.CrossEntropyLoss, 
        optimizer_cls=torch.optim.Adam,
        optimizer_args=dict(lr=5e-4, weight_decay=5e-4),
        epoch_scheduler_cls=torch.optim.lr_scheduler.StepLR, 
        epoch_scheduler_args=dict(step_size=10, gamma=0.2)
    )
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])
    cp.plot_metrics()
```

The training API should only be called once for the same checkpoint. Inside the training block, setup() can optionally be called, but no more than once. The setup() function contains instructions for how to instantiate the main components of the training process. When a component is provided, it is instantiated anew, resetting its state. If a component is not provided, the training loop uses the component and its end state from the previous checkpoint, if it exists. 

The train() function trains the model. It can be called multiple times. It is considered a cachable operation, i.e. the result of each train() function is cached based on the order of the train() calls. After the first execution of the checkpoint trainig block, the states of all components is stored, the results of all cachable operations are stored in order, and the state of the model parameters is stored. In subsequent executions, the components will be instantiated and their post-training state loaded, the model parameters will be loaded, and the execution of all cachable operations such as train() will be skipped and their output fetched from cache.

If we ran the example above, then added a third train() call and reran the block, the first two train() calls would be fetched from cache and the third train call would be executed as usual. However, if we ran the example above, then changed the setup() call to use SGD instead of Adam and reran the block, we would get an error since the saved state of the Adam optimizer contains parameters SGD does not accept. We would have to delete the checkpoint first using `model.delete_checkpoint(id=1)` and rerun the block.

## The Cache API

The cache API allows running cachable operations under a checkpoint, caching their results in its bundle of files.
```python
# Populate the cache
model.cache(id=1).find_lr(train_dataloader)
model.cache(id=1).evaluate(test_dataloader)
model.cache(id=2).find_lr(train_dataloader)

# Retrieve from cache
model.cache(id=1).find_lr(train_dataloader)

# Delete a cache entry
model.cache(id=1).delete('find_lr')

# Delete all cache entries
model.cache(id=1).delete()
```

The id argument of the cache() function is the checkpoint ID. Each checkpoint holds one cache slot per cachable operation that can be invoked through the Cache API. The first call populates the cache, subsequent calls retrieve from cache. Deleting a checkpoint using model.delete_checkpoint() deletes all cache entries in addition to the model parameters and training components.

## Loading prior checkpoints

In general, we can load model parameters and all training component to a prior checkpoint by calling load_checkpoint() on the model:
```python
model.load_checkpoint(id=1)
```

The Training API supports automatically reverting to the best performing model (as measured by the `watch` argument) at the end of the train() call by passing `load_best=True`. However, it does not revert the scheduler state by design. If we do:
```python
with model.checkpoint(1) as cp:
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])
```
then the second train() call continues with a lower learning rate immediately, starting with the parameters of the best performing model found during the first train() call. 

When using `load_best=False`, if the best performing model was observed prior to the very last training epoch, a backup of that model is created. It can be loaded using `model.load_checkpoint(id, from_backup=True)`:
```python
with model.checkpoint(id=2, description='Additional 40 epochs') as cp:
    # Because load_best=False, if the best performing model was observed in, let's say, epoch 25/40, two checkpoints are saved: "best" in epoch 25, and "last" in epoch 40. 
    cp.train(train_dl, test_dl, watch='accuracy', load_best=False, epochs=40, metrics=[train.metric_accuracy])
    cp.plot_metrics()

# At this point, the model reflects the "last" checkpoint at epoch 40.

# We can load the "best" from epoch 25 like so.
model.load_checkpoint(2, from_backup=True)
```

As before and unlike train(), load_checkpoint() reverts the schedulers to their state at the checkpoint by default. This can be controlled by passing `load_schedulers=False`:
```python
model.load_checkpoint(2, from_backup=True, load_schedulers=False)
```
