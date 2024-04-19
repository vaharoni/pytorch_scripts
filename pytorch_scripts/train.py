from contextlib import contextmanager
import numpy as np
import humanize
import torch
import os
import shutil
from tqdm.std import tqdm
from collections import defaultdict, namedtuple
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path
import pprint
from copy import deepcopy

root_folder = Path('..')
model_checkpoint_sep = '-'
noetbook_namespace = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_root_folder(root):
    global root_folder
    root_folder = Path(root)

def set_namespace(namespace):
    global noetbook_namespace
    noetbook_namespace = namespace

def set_model_checkpoint_sep(sep):
    global model_checkpoint_sep
    model_checkpoint_sep = str(sep)

def _ensure_namespace():
    if not noetbook_namespace:
        raise Exception('Must call train.set_namespace()')
    return True

def _ensure_no_space(id):
    if ' ' in id: 
        raise Exception(f'ID must not contain spaces (received "{id}")')
    return str(id)

_metric_factors = { 'loss': 1 }
def metric(name, higher_is_better=True):
    """Decorator args:
        name - the metric name. Will be attached to a prefix: train_{name} or val_{name}
        higher_is_better - whether higher or lower value of the metric is desirable.
    """
    def decorator(fn):
        fn.name = name
        fn.factor = -1 if higher_is_better else 1
        _metric_factors[name] = fn.factor
        return fn
    return decorator

# = Training helpers

@metric('accuracy')
def metric_accuracy(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()

@metric('binaccuracy')
def metric_binaccuracy(y_pred, y_true):
    return torch.sum(torch.round(y_pred) == y_true).item()

class EarlyStopException(Exception): pass

SetupLoss = namedtuple('SetupLoss', ['cls'])
SetupComponent = namedtuple('SetupComponent', ['cls', 'args', 'fn'])

class SetupManager():
    """ When setup() is called inside checkpoint blocks, we want the appropriate setup component (e.g. optimizer, scheduler) 
    to be built and overwritten on the model level. This includes the ability to partially override a component without affecting others - 
    an important consideration given that we want the state of optimizers and schedulers to be maintained.  
    When load_checkpoint() is called, we can rebuild the components from scratch, since the checkpoint file contains the appropriate 
    post-training states.
    """
    def __init__(self, model: 'Model') -> None:
        self.model = model
        self.component_recipes = {
            'loss': SetupLoss(None),
            'optimizer': SetupComponent(None, None, None),
            'epoch_scheduler': SetupComponent(None, None, None),
            'step_scheduler': SetupComponent(None, None, None),
        }

    def __repr__(self):
        return pprint.pformat(self.component_recipes, sort_dicts=False)

    def setup_loss(self, loss_cls=None):
        if loss_cls:
            self.component_recipes['loss'] = SetupLoss(loss_cls)
            self.build_loss()

    def _set_component(self, key, cls=None, args=None, fn=None, clear=False):
        if clear:
            self.component_recipes[key] = SetupComponent(None, None, None)
            return True
        if cls or args or fn:
            self.component_recipes[key] = SetupComponent(cls, args, fn)
            return True
        return False

    def setup_optimizer(self, optimizer_cls=None, optimizer_args=None, optimizer_fn=None):
        self._set_component('optimizer', optimizer_cls, optimizer_args, optimizer_fn) and self.build_optimizer()

    def setup_epoch_scheduler(self, epoch_scheduler_cls=None, epoch_scheduler_args=None, epoch_scheduler_fn=None, clear_epoch_scheduler=False):
        self._set_component('epoch_scheduler', epoch_scheduler_cls, epoch_scheduler_args, epoch_scheduler_fn, clear_epoch_scheduler) and self.build_scheduler('epoch_scheduler')

    def setup_step_scheduler(self, step_scheduler_cls=None, step_scheduler_args=None, step_scheduler_fn=None, clear_step_scheduler=False):
        self._set_component('step_scheduler', step_scheduler_cls, step_scheduler_args, step_scheduler_fn, clear_step_scheduler) and self.build_scheduler('step_scheduler')

    def build_loss(self):
        if self.component_recipes['loss'].cls:
            self.model.loss_fn = self.component_recipes['loss'].cls()

    def build_optimizer(self):
        component = self.component_recipes['optimizer']
        if component.fn:
            self.model.optimizer = component.fn(self.model.model.parameters())
        elif component.cls or component.args:
            optimizer_args = component.args if component.args else dict()
            self.model.optimizer = component.cls(self.model.model.parameters(), **optimizer_args)

    def build_scheduler(self, key):
        component = self.component_recipes[key]
        if component.fn:
            setattr(self.model, key, component.fn(self.model.optimizer))
        elif component.cls or component.args:
            scheduler_args = component.args if component.args else dict()
            setattr(self.model, key, component.cls(self.model.optimizer, **scheduler_args))
        else:
            setattr(self.model, key, None)

    def build_all(self):
        self.build_loss()
        self.build_optimizer()
        self.build_scheduler('epoch_scheduler')
        self.build_scheduler('step_scheduler')

    def clone(self):
        other = SetupManager(model=self.model)
        other.component_recipes = deepcopy(self.component_recipes)
        return other

# This is built with the idea of a callback. The constructor can take anything, while the hooks take a sensible API.
class MetricsManager:
    def __init__(self, 
                 metrics: list[callable], 
                 watch='loss', 
                 train_dataloader: DataLoader | None = None, 
                 val_dataloader: DataLoader | None = None, 
                 tensorboard_dir=None):
        self.history = defaultdict(lambda: [])
        self.metrics = metrics
        self.dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(tensorboard_dir)

        self.watch = watch
        self.watched_metric_factor = _metric_factors[self.watch]
        self.is_best_watched_metric_epoch = None
        self.best_watched_metric_epoch = None
        self.best_watched_metric = None
        self.global_epoch_start = None

    def log(self, msg):
        self.history['log'].append(msg)
        print(msg)

    def start_training(self, model: 'Model', epochs):
        self.global_epoch_start = model.epoch
        self.epochs = epochs
        self.best_watched_metric_epoch = max(self.global_epoch_start - 1, 0)
        mean_metrics = model.evaluate(self.dataloaders['val'], metrics=self.metrics)
        self.best_watched_metric = mean_metrics[self.watch]

        self.mean_metrics = { f'val_{metric_name}': metric_val for metric_name, metric_val in mean_metrics.items() }
        metrics_str = ", ".join([f'val_{metric_name}={metric_val:.4f}' for metric_name, metric_val in mean_metrics.items()])
        self.log(f'Initial {metrics_str}')

        if self.global_epoch_start == 0:
            inputs, _ = next(iter(self.dataloaders['val']))
            self.tb_writer.add_graph(model.model, inputs.to(device))
            self.tb_writer.flush()

    def start_step(self, phase, local_step):
        pass
        
    def start_epoch(self, local_epoch):
        self.is_best_watched_metric_epoch = False
        self.running_metrics = { 'train': defaultdict(lambda: 0), 'val': defaultdict(lambda: 0) }

    def end_step(self, phase, local_step, total_loss, y_pred, y_true):
        self.running_metrics[phase]['loss'] += total_loss
        for metric in self.metrics:
            metric_val = metric(y_pred, y_true)
            self.running_metrics[phase][metric.name] += metric_val

    def end_epoch(self, local_epoch):
        self.mean_metrics = { f"{phase}_{metric_name}": self.running_metrics[phase][metric_name] / len(self.dataloaders[phase].dataset)
                                for phase in self.running_metrics  
                                    for metric_name in self.running_metrics[phase] }

        # Print loss and metrics
        self.log(f"Epoch {local_epoch + 1}/{self.epochs}: " + ", ".join([f"{metric_name}={self.mean_metrics[metric_name]:.4f}" for metric_name in self.mean_metrics]))

        # Store metrics in history
        self.history['epoch'].append(self.global_epoch_start + local_epoch + 1)
        for full_metric_name in self.mean_metrics:
            self.history[full_metric_name].append(self.mean_metrics[full_metric_name])

        # Report to tensorboard
        if self.tb_writer is not None:
            tb_metrics = defaultdict(lambda: dict())
            for full_metric_name in self.mean_metrics:
                phase, *metric_name = full_metric_name.split('_')
                metric_name = "_".join(metric_name)
                tb_metrics[metric_name][phase] = self.mean_metrics[full_metric_name]
            for metric_name in tb_metrics:
                self.tb_writer.add_scalars(metric_name, tb_metrics[metric_name], self.global_epoch_start + local_epoch + 1)
            self.tb_writer.flush()

        # Check if we exceeded the best watched metric
        if self.mean_metrics[f'val_{self.watch}'] * self.watched_metric_factor < self.best_watched_metric * self.watched_metric_factor:
            self.is_best_watched_metric_epoch = True
            self.best_watched_metric_epoch = self.global_epoch_start + local_epoch
            self.best_watched_metric = self.mean_metrics[f'val_{self.watch}']

class WarmUpLR(torch.optim.lr_scheduler.LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class Model:
    def __init__(self, id, model: torch.nn.Module, description=None):
        _ensure_namespace()
        self.id = _ensure_no_space(id)
        self.description = description
        self.model = model.to(device)
        self.setup_manager = SetupManager(model=self)
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = None
        self.epoch_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.step_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.step = 0
        self.epoch = 0
        self.checkpoints = {}
        self.active_checkpoint = None
        self.print_params(False)

    def tensorboard_path(self) -> Path:
        return root_folder / 'runs' / noetbook_namespace / self.id
    
    def checkpoint_path(self) -> Path:
        return root_folder / 'ckpts' / noetbook_namespace / self.id
    
    def _obtain_checkpoint(self, id, description=None) -> 'CheckpointLifecycle':
        cp = self.checkpoints.get(id)
        if not cp:
            cp = CheckpointLifecycle(model=self, id=id, description=description)
            self.checkpoints[id] = cp
        return cp
    
    def _remove_checkpoint(self, id):
        cp = self._obtain_checkpoint(id)
        if id in self.checkpoints:
            del self.checkpoints[id]
        if self.active_checkpoint and self.active_checkpoint.id == id:
            self.active_checkpoint = None
        return cp
    
    def set_epoch_and_step(self, global_epoch, global_step):
        self.epoch = global_epoch
        self.step = global_step

    def setup(self, 
              loss_cls=None, 
              optimizer_cls=None, optimizer_args=None, optimizer_fn=None, 
              epoch_scheduler_cls=None, epoch_scheduler_args=None, epoch_scheduler_fn=None, clear_epoch_scheduler=False,
              step_scheduler_cls=None, step_scheduler_args=None, step_scheduler_fn=None, clear_step_scheduler=False):
        self.setup_manager.setup_loss(loss_cls)
        self.setup_manager.setup_optimizer(optimizer_cls, optimizer_args, optimizer_fn)
        self.setup_manager.setup_epoch_scheduler(epoch_scheduler_cls, epoch_scheduler_args, epoch_scheduler_fn, clear_epoch_scheduler)
        self.setup_manager.setup_step_scheduler(step_scheduler_cls, step_scheduler_args, step_scheduler_fn, clear_step_scheduler)

    def serialize(self):
        data = {
            'model_id': self.id, 
            'model_description': self.description,
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
        }
        if self.optimizer:
            data['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.epoch_scheduler:
            data['epoch_scheduler_state_dict'] = self.epoch_scheduler.state_dict()
        if self.step_scheduler:
            data['step_scheduler_state_dict'] = self.step_scheduler.state_dict()
        return data
    
    def deserialize(self, data, load_schedulers=True):
        if not self.description:
            self.description = data['model_description']
        self.step = data['step']
        self.epoch = data['epoch']
        self.model.load_state_dict(data['model_state_dict'])
        if data.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(data['optimizer_state_dict'])
        if load_schedulers and data.get('epoch_scheduler_state_dict'):
            self.epoch_scheduler.load_state_dict(data['epoch_scheduler_state_dict'])
        if load_schedulers and data.get('step_scheduler_state_dict'):
            self.step_scheduler.load_state_dict(data['step_scheduler_state_dict'])

    @contextmanager
    def checkpoint(self, id, description=None):
        cp = self._obtain_checkpoint(id, description)
        self.active_checkpoint = cp
        yield CheckpointTrainAPI(cp)

    def cache(self, id, description=None):
        cp = self._obtain_checkpoint(id, description)
        self.active_checkpoint = cp
        return CheckpointCacheAPI(cp)

    def load_checkpoint(self, id, from_backup=False, load_schedulers=True):
        cp = self._obtain_checkpoint(id)
        self.active_checkpoint = cp
        self.setup_manager = cp.setup_manager_copy.clone()
        self.setup_manager.build_all()
        cp.load(backup=from_backup, load_schedulers=load_schedulers, logger=print)

    def delete_checkpoint(self, id):
        cp = self._remove_checkpoint(id)
        cp.delete()

    def delete(self):
        shutil.rmtree(self.tensorboard_path(), ignore_errors=True)
        shutil.rmtree(self.checkpoint_path(), ignore_errors=True)
        self.checkpoints = {}
        self.active_checkpoint = None        

    def find_lr(self, dataloader, optimizer_cls=None, optimizer_args=None, loss_cls=None, epochs=1, min_rate=1e-8, max_rate=1):
        """ Typical usage, which will clone the model's optimizer and its internal state:

            res = model.find_lr(train_dl)
            res.plot()
        """
        if loss_cls:
            loss_fn = loss_cls()
        elif self.loss_fn:
            loss_fn = self.loss_fn
        else:
            raise Exception('The model does not have a loss function and none was provided')
        
        if optimizer_cls or optimizer_args:
            optimizer_args = optimizer_args if optimizer_args else dict()
            optimizer_args = {**optimizer_args, 'lr': min_rate}
            optimizer: torch.optim.Optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        elif self.optimizer:
            optimizer: torch.optim.Optimizer = self.optimizer.__class__(self.model.parameters())
            optimizer.load_state_dict(self.optimizer.state_dict())
            for param in optimizer.param_groups:
                param['lr'] = min_rate
        else:
            raise Exception('The model does not have an optimizer and none was provided')
        
        if not self.active_checkpoint:
            orig_state = deepcopy(self.model.state_dict())

        iterations = len(dataloader) * epochs
        gamma = (max_rate / min_rate) ** (1 / iterations)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        self.model.train()
        rates, losses = [], []
        for _ in range(epochs):
            for inputs, y_true in tqdm(dataloader):
                inputs = inputs.to(device)
                y_true = y_true.to(device)
                optimizer.zero_grad()
                y_pred = self.model(inputs)
                loss: torch.Tensor = loss_fn(y_pred, y_true)
                rates.append(optimizer.param_groups[0]['lr'])
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                
        if self.active_checkpoint:
            self.active_checkpoint.load()
        else:
            self.model.load_state_dict(orig_state)

        return LearningRateVsLoss(rates, losses)

    def train(self, 
              train_dataloader: DataLoader, 
              val_dataloader: DataLoader, 
              watch: str = 'loss', 
              load_best: bool = True,
              epochs: int = 10, 
              patience: int = 0, 
              warmup: int = 0, 
              metrics: list[callable] = [],
              tensorboard_dir: str = None,
              save_checkpoint_fn: callable = None,
              save_hist_fn: callable = None,            # Separate from checkpoint, since we want to record the full history even it we revert to best prior state
              load_best_fn: callable = None):

        dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

        metrics_manager = MetricsManager(metrics=metrics, train_dataloader=train_dataloader, val_dataloader=val_dataloader, watch=watch, tensorboard_dir=tensorboard_dir)
        metrics_manager.start_training(model=self, epochs=epochs)

        global_epoch_start = self.epoch
        global_step_start = self.step
        local_step = 0

        # Creating a checkpoint file prior to training so that the file structure is always consistent with the post-trained model in memory,
        # and so that the load_best_fn() reverts to it in case no improvement is observed during training and load_best is True
        if save_checkpoint_fn:
            save_checkpoint_fn(global_epoch_start, global_step_start, metrics_manager.mean_metrics, make_backup=False, logger=metrics_manager.log)

        steps_per_epoch = len(train_dataloader)
        warmup_scheduler = None

        if warmup > 0:
            warmup_scheduler = WarmUpLR(self.optimizer, total_iters=steps_per_epoch * warmup)

        try: 
            for local_epoch in range(epochs):
                metrics_manager.start_epoch(local_epoch)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    # Iterate over data.
                    for inputs, y_true in tqdm(dataloaders[phase]):
                        inputs: torch.Tensor = inputs.to(device)
                        y_true: torch.Tensor = y_true.to(device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            y_pred = self.model(inputs)
                            loss: torch.Tensor = self.loss_fn(y_pred, y_true)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                if warmup_scheduler and warmup > local_epoch:
                                    warmup_scheduler.step()

                        metrics_manager.end_step(phase=phase, local_step=local_step, total_loss=loss.item() * inputs.shape[0], y_pred=y_pred, y_true=y_true)

                        if phase == 'train':
                            local_step += 1
                            # Advance the step LR scheduler
                            if self.step_scheduler is not None and warmup <= global_epoch_start + local_epoch:
                                self.step_scheduler.step()

                    # Advance the epoch LR scheduler
                    if phase == 'train' and self.epoch_scheduler is not None and warmup <= global_epoch_start + local_epoch:
                        self.epoch_scheduler.step()

                metrics_manager.end_epoch(local_epoch=local_epoch)

                if save_hist_fn:
                    save_hist_fn(metrics_manager.history)

                if metrics_manager.is_best_watched_metric_epoch and save_checkpoint_fn:
                    save_checkpoint_fn(global_epoch_start + local_epoch + 1, global_step_start + local_step, metrics_manager.mean_metrics, make_backup=False, logger=metrics_manager.log)

                if patience > 0 and global_epoch_start + local_epoch - metrics_manager.best_watched_metric_epoch >= patience:
                    raise EarlyStopException
                
        except EarlyStopException:
            pass

        if load_best:
            if load_best_fn:
                load_best_fn(logger=metrics_manager.log)
        else:
            if save_checkpoint_fn and not metrics_manager.is_best_watched_metric_epoch:
                save_checkpoint_fn(global_epoch_start + local_epoch + 1, global_step_start + local_step, metrics_manager.mean_metrics, make_backup=True, logger=metrics_manager.log)

        if save_hist_fn:
            save_hist_fn(metrics_manager.history)
                
        return metrics_manager.history


    def evaluate(self, dataloader: DataLoader, loss_cls=None, metrics: list[callable] = []):
        if loss_cls:
            loss_fn = loss_cls()
        elif self.loss_fn:
            loss_fn = self.loss_fn
        else:
            raise Exception('The model does not have a loss function and none was provided')
            
        self.model.eval()
        running_metrics = defaultdict(lambda: 0)
        with torch.no_grad():
            for inputs, y_true in dataloader:
                inputs: torch.Tensor = inputs.to(device)
                y_true: torch.Tensor = y_true.to(device)

                y_pred = self.model(inputs)
                loss = loss_fn(y_pred, y_true)
                running_metrics['loss'] += loss.item() * inputs.shape[0]
                for metric in metrics:
                    metric_val = metric(y_pred, y_true)
                    running_metrics[metric.name] += metric_val
            
            mean_metrics = { metric_name: running_metrics[metric_name] / len(dataloader.dataset)
                            for metric_name in running_metrics }
            
            return mean_metrics
        
    def print_params(self, full=True):
        stats = defaultdict(lambda: dict())
        trainable = 0
        untrainable = 0
        buffers = 0
        human_fn = humanize.metric 

        for full_name, param in self.model.named_parameters():
            *layer_name, param_name = full_name.split('.')
            layer_name = '.'.join(layer_name)
            param_count = np.prod(param.shape)
            stats[layer_name][param_name] = param
            if param.requires_grad: 
                trainable += param_count
            else:
                untrainable += param_count

        for full_name, buff in self.model.named_buffers():
            *layer_name, buff_name = full_name.split('.')
            layer_name = '.'.join(layer_name)
            param_count = np.prod(param.shape)
            stats[layer_name][buff_name] = buff
            buffers += param_count

        longest_layer_name = np.max([len(x) for x in stats.keys()])
        longest_param_or_buff_name = np.max([len(x) for l in stats for x in stats[l].keys()])

        if full:
            for layer_name in stats:
                for i, param_or_buff_name in enumerate(stats[layer_name]):
                    param_or_buff = stats[layer_name][param_or_buff_name]
                    param_count = np.prod(param_or_buff.shape)
                    if isinstance(param_or_buff, torch.nn.Parameter):
                        type_char = '✅' if param_or_buff.requires_grad else '🔒'
                    else:
                        type_char = '⛔'
                    if i == 0:
                        print_name = f'{layer_name:{longest_layer_name}}'
                    else:
                        pad = ' '
                        print_name = f'{pad:{longest_layer_name}}'

                    print(print_name, f'{param_or_buff_name:{longest_param_or_buff_name}}', type_char, '{:10}'.format(human_fn(param_count)), list(param_or_buff.shape))

            print('-' * 100)        
            print(f"✅ Trainable params:   {human_fn(trainable)}")
            print(f"🔒 Untrainable params: {human_fn(untrainable)}")
            print(f"⛔ Buffers:            {human_fn(buffers)}")
            print('-' * 100)
        else:
            print(f"Trainable params: {human_fn(trainable)}. Untrainable params: {human_fn(untrainable)}. Buffers: {human_fn(buffers)}.")

class CacheFile:
    def __init__(self, file_path) -> None:
        self.file_path = Path(file_path)
        self.data = None
        self._load()

    def __repr__(self) -> str:
        if not self.data:
            return None.__repr__()
        if isinstance(self.data, list):
            return { 'count': len(self.data) }.__repr__()
        if isinstance(self.data, dict):
            return {k: True for k in self.data}.__repr__()
    
    def get_from_list(self, index=None):
        data = self.data or []
        if not isinstance(data, list):
            raise Exception('get_from_list() can only be called for list caches')
        
        if index is not None:
            if data and index < len(data):
                return data[index]
        else:
            return data

    def get_from_dict(self, key=None):
        data = self.data or {}
        if key is not None:
            return data.get(key)
        else:
            return data
        
    def set_list_item(self, index, data):
        if self.data is None:
            self.data = []
        elif not isinstance(self.data, list):
            raise Exception('set_list_item() can only be called for list caches')
            
        curr_length = len(self.data)
        if index < curr_length:
            self.data[index] = data
        elif index == curr_length:
            self.data.append(data)
        else:
            raise Exception(f'Cannot set list item at index {index} for list length of {curr_length}')
        
    def set_dict_item(self, key, data):
        if self.data is None:
            self.data = dict()
        elif not isinstance(self.data, dict):
            raise Exception('set_dict_item() can only be called for dict caches')
        self.data[key] = data

    def remove_dict_item(self, key=None):
        if self.data is None:
            return
        elif not isinstance(self.data, dict):
            raise Exception('set_dict_item() can only be called for dict caches')
        if key:
            del(self.data[key])
        else:
            self.data = None

    def _load(self):
        if not self.file_path.exists():
            return False
        self.data = torch.load(self.file_path)
        return True
    
    def save(self):
        if not self.data:
            self.delete()
            return False
        os.makedirs(self.file_path.parent, exist_ok=True)
        torch.save(self.data, self.file_path)
        return True

    def delete(self):
        if not self.file_path.exists():
            return False
        os.remove(self.file_path)
        return True

class CheckpointLifecycle:
    def __init__(self, model: Model, id, description=None):
        self.model = model
        self.id = id
        self.norm_id = self._calc_norm_id(id)
        self.description = description
        self.persisted = False
        self.metrics = {}
        cache_prefix = str(self.model.checkpoint_path() / (self.norm_id + model_checkpoint_sep))
        self.train_api_cache = CacheFile(cache_prefix + 'train.cache')
        self.cache_api_cache = CacheFile(cache_prefix + 'cache.cache')
        self.setup_manager_copy = model.setup_manager.clone()

    def _calc_norm_id(self, id):
        if isinstance(id, int):
            return str(id).zfill(4) 
        elif isinstance(id, float):
            ip = str(math.floor(id)).zfill(4)
            fp = str(math.floor(id % 1 * 100))
            return ip + model_checkpoint_sep + fp
        else:
            return _ensure_no_space(id)

    def __repr__(self):
        data = {
            'model_id': self.model.id,
            'model_description': self.model.description,
            'checkpoint_id': self.id,
            'checkpoint_norm_id': self.norm_id,
            'checkpoint_description': self.description,
            'persisted': self.persisted,
            'metrics': self.metrics,
            'setup': self.setup_manager_copy,
            'train_api_cache': self.train_api_cache,
            'cache_api_cache': self.cache_api_cache
        }
        return pprint.pformat(data, sort_dicts=False)

    def setup(self, *args, **opts):
        self.model.setup(*args, **opts)
        self.setup_manager_copy = self.model.setup_manager.clone()

    def serialize(self):
        data = {
            **self.model.serialize(),
            'checkpoint_id': self.id,
            'checkpoint_norm_id': self.norm_id,
            'checkpoint_description': self.description,
            'checkpoint_metrics': self.metrics
        }
        return data

    def deserialize(self, data, load_schedulers=True):
        self.model.deserialize(data, load_schedulers=load_schedulers)
        if not self.description:
            self.description = data['checkpoint_description']
        self.metrics = data['checkpoint_metrics']

    def tensorboard_path(self) -> Path:
        return self.model.tensorboard_path() / self.norm_id
    
    def checkpoint_path(self, backup=False) -> Path:
        version = 'best' if backup else 'last'
        return self.model.checkpoint_path() / f'{self.norm_id + model_checkpoint_sep + version}.pt'
    
    # The knowledge of whether a backup exists is only present in the calling code. There is no trace for such information in the data stored.
    def load(self, backup=False, load_schedulers=True, logger=None):
        path = self.checkpoint_path(backup=backup)

        if backup and not path.exists():
            if logger:
                logger(f"Backup for model {self.model.id} checkpoint {self.id} does not exist. Trying from latest.")
            path = self.checkpoint_path(backup=False)
            
        if not path.exists(): 
            if logger:
                logger(f"Cannot find checkpoint {self.id} for model {self.model.id}")
            return False

        data = torch.load(path)
        self.deserialize(data, load_schedulers=load_schedulers)
        self.persisted = True

        if logger:
            loaded_str = f"Loaded model {self.model.id} from checkpoint {self.id}. epoch={self.model.epoch}, step={self.model.step}."
            metrics_str = ", ".join([f'{metric_name}={metric_val:.4f}' for metric_name, metric_val in self.metrics.items()])
            logger(" ".join([x for x in [loaded_str, metrics_str] if x]))

        return True

    def save(self, global_epoch, global_step, metrics, make_backup=False, logger=None):
        save_path = self.checkpoint_path(backup=False)
        os.makedirs(save_path.parent, exist_ok=True)

        if self.persisted and make_backup:
            backup_path = self.checkpoint_path(backup=True)
            os.rename(save_path, backup_path)
            logger(f"Created backup for model {self.model.id} checkpoint {self.id}.")

        self.model.set_epoch_and_step(global_epoch, global_step)
        self.metrics = dict(metrics)
        data = self.serialize()
        torch.save(data, save_path)
        self.persisted = True

        if logger:
            saved_str = f"Saved model {self.model.id} checkpoint {self.id}. epoch={global_epoch}, step={global_step}."
            metrics_str = ", ".join([f'{metric_name}={metric_val:.4f}' for metric_name, metric_val in self.metrics.items()])
            logger(" ".join([x for x in [saved_str, metrics_str] if x]))

        return True
            
    def combined_history(self):
        padding = 0
        hist_dict = defaultdict(lambda: [None] * padding)
        for hist in self.train_api_cache.get_from_list():
            for key, values in hist.items():
                hist_dict[key] += values
            padding += len(hist['epoch'])
        return dict(hist_dict)

    def delete(self):
        backup_path = self.checkpoint_path(backup=True)
        if backup_path.exists():
            os.remove(backup_path)

        save_path = self.checkpoint_path(backup=False)
        if save_path.exists():
            os.remove(save_path)

        self.train_api_cache.delete()
        self.cache_api_cache.delete()
        self.persisted = False

class CheckpointCacheAPI:
    def __init__(self, cp_lifecycle: CheckpointLifecycle) -> None:
        self._cp_lifecycle = cp_lifecycle

    def __repr__(self) -> str:
        return self._cp_lifecycle.__repr__()
    
    def _cache(self, key, calculate_fn: callable):
        data = self._cp_lifecycle.cache_api_cache.get_from_dict(key)
        if data:
            return data        
        res = calculate_fn()
        self._cp_lifecycle.cache_api_cache.set_dict_item(key, res)
        self._cp_lifecycle.cache_api_cache.save()
        return res
        
    def evaluate(self, *args, **opts):
        return self._cache('evaluate', lambda: self._cp_lifecycle.model.evaluate(*args, **opts))

    def find_lr(self, *args, **opts):
        return self._cache('find_lr', lambda: self._cp_lifecycle.model.find_lr(*args, **opts))
    
    def delete(self, key=None):
        if key:
            self._cp_lifecycle.cache_api_cache.remove_dict_item(key)
            self._cp_lifecycle.cache_api_cache.save()
        else:
            self._cp_lifecycle.cache_api_cache.remove_dict_item()
            self._cp_lifecycle.cache_api_cache.delete()

class CheckpointTrainAPI:
    def __init__(self, cp_lifecycle: CheckpointLifecycle):
        self._cp_lifecycle = cp_lifecycle
        self._setup_called = False
        self._train_method_call_count = 0

    def setup(self, *args, **opts):
        """args:
            Loss factory:
                loss_cls=torch.nn.CrossEntropyLoss,

            Optimizer factory, option 1:
                optimizer_cls=torch.optim.Adam
                optimizer_args=dict(weight_decay=5e-4, lr=0.001)

            Optimizer factory, option 2:
                optimizer_fn=lambda params: torch.optim.Adam(params, weight_decay=54-4, lr=0.001)

            Epoch scheduler, option 1:
                epoch_scheduler_cls=torch.optim.lr_scheduler.StepLR
                epoch_scheduler_args=dict(step_size=40, gamma=0.2)

            Epoch scheduler, option 2:
                epoch_scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)

            Step scheduler, option 1:
                step_scheduler_cls=torch.optim.lr_scheduler.StepLR
                step_scheduler_args=dict(step_size=40, gamma=0.2)

            Step scheduler, option 2:
                step_scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
        """         
        if self._setup_called:
            raise Exception('setup() can only be called once per checkpoint')
        self._setup_called = True
        self._cp_lifecycle.setup(*args, **opts)
        return self

    def train(self, 
              train_dataloader: DataLoader, val_dataloader: DataLoader, 
              watch: str = 'loss', load_best: bool = True,
              epochs: int = 10, 
              patience: int = 0, 
              warmup: int = 0, 
              metrics: list[callable] = []):
        
        if self._train_method_call_count == 0:
            self._cp_lifecycle.load()

        self._train_method_call_count += 1

        hist = self._cp_lifecycle.train_api_cache.get_from_list(self._train_method_call_count - 1)
        if hist:
            for msg in hist['log']:
                print(msg)
            return hist

        tensorboard_dir = self._cp_lifecycle.tensorboard_path()

        def save_checkpoint_fn(*args, **opts):
            self._cp_lifecycle.save(*args, **opts)

        def save_hist_fn(hist):
            self._cp_lifecycle.train_api_cache.set_list_item(self._train_method_call_count - 1, dict(hist))
            self._cp_lifecycle.train_api_cache.save()

        def load_best_fn(logger=None):
            self._cp_lifecycle.load(backup=False, load_schedulers=False, logger=logger)

        return self._cp_lifecycle.model.train(train_dataloader=train_dataloader, 
                                              val_dataloader=val_dataloader, 
                                              watch=watch, 
                                              load_best=load_best, 
                                              epochs=epochs, 
                                              patience=patience, 
                                              warmup=warmup, 
                                              metrics=metrics, 
                                              tensorboard_dir=tensorboard_dir, 
                                              save_checkpoint_fn=save_checkpoint_fn,
                                              save_hist_fn=save_hist_fn,
                                              load_best_fn=load_best_fn)
    
    def plot_metrics(self, hist=None):
        hist = hist if hist else self._cp_lifecycle.combined_history()
        plot_metrics(hist)

class LearningRateVsLoss:
    def __init__(self, rates, losses) -> None:
        self.rates = np.array(rates)
        self.losses = np.array(losses)

    def plot(self, start=None, end=None):
        start = np.argmax(self.rates >= start) if start else 0
        end = np.argmax(self.rates > end) if end else None
        end = None if end == 0 else end
        plot_lr_vs_loss(self.rates[start:end], self.losses[start:end])

def plot_metrics(hist):
    x = hist['epoch']
    y = defaultdict(lambda: {})
    for metric_name in hist:
        if metric_name in ['epoch', 'log']:
            continue
        chart_name = "_".join(metric_name.split('_')[1:])
        y[chart_name][metric_name] = hist[metric_name]
    charts = len(y)
    rows = math.ceil(charts / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
    axes = axes.ravel()
    for i, chart_name in enumerate(y):
        ax: plt.Axes = axes[i]
        ax.set_title(chart_name)
        for metric_name in y[chart_name]:
            ax.plot(x, hist[metric_name], label=metric_name)
        ax.legend(loc='upper left')
        ax.set_xticks(x)
        ax.grid()

def plot_lr_vs_loss(rates, losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx()
    ax.plot(rates, losses, "b")
    ax.hlines(min(losses), min(rates), max(rates), color="k")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.grid()