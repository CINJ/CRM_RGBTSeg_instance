# coding:utf-8
# Written by Ukcheol Shin, Jan. 24, 2023
# Email: shinwc159@gmail.com

import os
import copy
import itertools
import time
import torch
import torch.nn as nn 
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import torchvision.utils as vutils
from torchvision.ops import nms
import json
from util.util import get_palette_PV1
from util.torch_tensor_encoder import TorchTensorEncoder
from .registry import MODELS
from models.mask2former import RGBTMaskFormer
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.structures.boxes import Boxes
import cv2
import numpy as np


@MODELS.register_module(name='RGBTMaskFormer')
class Model_RGBT_Mask2Former(LightningModule):
    def __init__(self, cfg):
        super(Model_RGBT_Mask2Former, self).__init__()
        self.save_hyperparameters()
        self.max_loss = 0
        self.counter = 0

        # self.cfg = cfg
        self.output_dir = cfg.SAVE.DIR_ROOT
        self.metrics_filename = os.path.join(self.output_dir,time.strftime("log_%Y%m%d_%H%M%S.txt"))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "thr"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "pred"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "gt"), exist_ok=True)
        
        # Set our init args as class attributes
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.lr_decay = cfg.SOLVER.WEIGHT_DECAY

        self.label_list = ["unlabelled", "penguin"]
        self.palette = get_palette_PV1()

        self.network = RGBTMaskFormer(cfg)
        self.optimizer = self.build_optimizer(cfg, self.network)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.automatic_optimization = False

        self.w_mws = cfg.MODEL.CRMLOSS.MWS_WEIGHT
        self.w_sdc = cfg.MODEL.CRMLOSS.SDC_WEIGHT 
        self.w_sdn = cfg.MODEL.CRMLOSS.SDN_WEIGHT 

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def forward(self, x):
        logits = self.network(x)
        return logits.argmax(1).squeeze()

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim = self.optimizers()

        # tensorboard logger
        logger = self.logger.experiment

        # get network output
        losses_dict = self.network(batch_data)
        # print("LOSSES DICT")
        # for item in losses_dict:
        #     print(str(type(item)))
        #     print(item)
        #     print("----")
        #     print()
        # print()
        # print()
        
        loss = sum(losses_dict[0].values())
        loss += self.w_mws*sum(losses_dict[1].values()) # RGB
        loss += self.w_mws*sum(losses_dict[2].values()) # THR 
        loss += sum(losses_dict[3].values()) # Masked RGB-T
        loss += self.w_sdc*losses_dict[4] # self-distillation for complementary representation  
        loss += self.w_sdn*losses_dict[5] # self-distillation for non-local representation 

        # optimize network
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        # log
        self.log('train/total_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        scheduler = self.lr_schedulers()
        scheduler.step()
            
    def __create_visualization_for_instances(self, batch_instances, image_shape, ground_truth):
        # Initialize an empty tensor with the shape [batch_size, H, W]
        merged_masks = torch.zeros((len(batch_instances),1) + tuple(image_shape[-2:]), dtype=torch.uint8)
        for batch_idx, instance in enumerate(batch_instances):
            masks = instance.pred_masks if not ground_truth else instance.gt_masks
            for i, mask in enumerate(masks):
                merged_masks[batch_idx,0][mask > 0] = i + 1  # Assign a unique label for each instance        
        return merged_masks

    def __binary_flatten(self, masks):
        merged_masks = torch.zeros(tuple(masks.shape[-2:]), dtype=torch.uint8)
        for i, mask in enumerate(masks):
            merged_masks[mask > 0] = 1  # Assign a unique label for each instance        
        return merged_masks

    def __compute_precision_recall_loss_confusion(self, gt_instances, pred_instances):
        # print(vars(gt_instances))
        # print(vars(pred_instances))
        flat_gt_mask = self.__binary_flatten(gt_instances.gt_masks.bool()).flatten()
        flat_pred_mask = self.__binary_flatten(pred_instances.pred_masks.bool()).flatten()
        TP = torch.sum((flat_gt_mask == 1) & (flat_pred_mask == 1)).float()
        FP = torch.sum((flat_gt_mask == 0) & (flat_pred_mask == 1)).float()
        FN = torch.sum((flat_gt_mask == 1) & (flat_pred_mask == 0)).float()
        TN = torch.sum((flat_gt_mask == 0) & (flat_pred_mask == 0)).float()
        precision = TP / (TP + FP + 1e-6)  # Adding a small epsilon to avoid division by zero
        recall = TP / (TP + FN + 1e-6)
        loss = F.binary_cross_entropy(flat_pred_mask.float(), flat_gt_mask.float())
        return precision, recall, loss, TP, TN, FP, FN
        
    def __compute_iou_for_instances(self, gt_instances, pred_instances):
        gt_masks = gt_instances.gt_masks.bool()
        pred_masks = pred_instances.pred_masks.bool()
        ious = []
        for gt_mask in gt_masks:
            best_iou = 0.0
            for pred_mask in pred_masks:
                intersection = (gt_mask & pred_mask).float().sum()
                union = (gt_mask | pred_mask).float().sum()
                iou = (intersection / union).item() if union > 0 else 0.0
                best_iou = max(best_iou, iou)
            ious.append(best_iou)
        return ious
        
    def __package_result(self, iou, precision, recall, loss, tp, tn, fp, fn, ca, commit=True) -> {}:
        if (self.max_loss == 0) or (loss > self.max_loss):
            self.max_loss = loss
            scaled_loss = 1.0
        else:
            scaled_loss = loss / self.max_loss
        output = {"iou": iou, "precision": precision, "recall": recall, "loss": scaled_loss, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "ca": ca}
        if commit:
            with open(self.metrics_filename, 'a') as file:
                file.write(json.dumps(output, cls=TorchTensorEncoder) + '\n')
        return output

    def __nms(self, pred_instances, iou_threshold):
        # print(str(type(pred_instances.pred_boxes)))
        # print(str(type(pred_instances.scores)))
        # print(str(type(pred_instances.pred_classes)))
        device = pred_instances.pred_boxes.device  # Get the device (CPU or CUDA)
        if type(pred_instances.pred_boxes) is Boxes:
            boxes = pred_instances.pred_boxes.tensor.to(device)
        else:
            boxes = pred_instances.pred_boxes.to(device)
        scores = pred_instances.scores.to(device)
        labels = pred_instances.pred_classes.to(device)
        keep = nms(boxes, scores, iou_threshold)
        pred_instances.pred_boxes = boxes[keep]
        pred_instances.scores = scores[keep]
        pred_instances.pred_classes = labels[keep]
        pred_instances.pred_masks = pred_instances.pred_masks[keep]  # Include masks if needed            
        return pred_instances

    def __count_instances(self, pred_instances, gt_instances, class_id=1, iou_threshold=0.5):
        pred_instances = [self.__nms(pred, iou_threshold) for pred in pred_instances]
        pred_count = sum((pred.pred_classes == class_id).sum().item() for pred in pred_instances)
        gt_count = sum((gt.gt_classes == class_id).sum().item() for gt in gt_instances)
        return pred_count, gt_count
    
    def validation_step(self, batch_data, batch_idx):
        images = [x["image"] for x in batch_data]
        gt_instances = [x["instances"] for x in batch_data]
        logger = self.logger.experiment
        logits = self.network(batch_data)
        pred_instances = [x["instances"] for x in logits]
        images = torch.stack(images) 
        ious = []
        precision = []
        recall = []
        loss = []
        tp = [] 
        tn = []
        fp = []
        fn = []
        ca = []
        for gt_inst, pred_inst in zip(gt_instances, pred_instances):
            pred_count, gt_count = self.__count_instances(pred_instances, gt_instances)
            ca.append( 1 - torch.abs(torch.tensor(pred_count) - torch.tensor(gt_count)) / (torch.tensor(gt_count) + 1e-6))
            iou = self.__compute_iou_for_instances(gt_inst, pred_inst)
            ious.append(torch.tensor(iou).mean())
            p,r,l, TP, TN, FP, FN = self.__compute_precision_recall_loss_confusion(gt_inst, pred_inst)
            precision.append(p)
            recall.append(r)
            loss.append(l)
            tp.append(TP)
            tn.append(TN)
            fp.append(FP)
            fn.append(FN)
        mean_iou = torch.tensor(ious).mean()
        mean_precision = torch.tensor(precision).mean()
        mean_recall = torch.tensor(recall).mean()
        mean_loss = torch.tensor(loss).mean()
        mean_tp = torch.tensor(tp).mean()
        mean_tn = torch.tensor(tn).mean()
        mean_fp = torch.tensor(fp).mean()
        mean_fn = torch.tensor(fn).mean()
        mean_ca = torch.tensor(ca).mean()
    
        if batch_idx % 100 == 0:
            # Visualization similar to the semantic segmentation example, but now for instance masks
            rgb_vis = (images[:, :3, ...]).type(torch.uint8)
            thr_vis = (images[:,[-1],...]).type(torch.uint8)
            thr_vis  = torch.cat((thr_vis, thr_vis, thr_vis), 1) 
            pred_vis = self.__create_visualization_for_instances(pred_instances, rgb_vis.shape, False)  # Custom function for visualization
            pred_vis  = torch.cat((pred_vis, pred_vis, pred_vis), 1) 
            gt_vis = self.__create_visualization_for_instances(gt_instances, rgb_vis.shape, True)  # Custom function for visualization
            gt_vis  = torch.cat((gt_vis, gt_vis, gt_vis), 1) 
            
            input_rgb_grid = vutils.make_grid(rgb_vis, nrow=8, padding=10, pad_value=1.0)
            input_thr_grid = vutils.make_grid(thr_vis, nrow=8, padding=10, pad_value=1.0)
            gt_grid = vutils.make_grid(gt_vis, nrow=8, padding=10, pad_value=1.0)
            prediction_grid = vutils.make_grid(pred_vis, nrow=8, padding=10, pad_value=1.0)
            result_grid = torch.cat((input_rgb_grid.cpu().detach(), input_thr_grid.cpu().detach(), gt_grid.cpu().detach(), prediction_grid.cpu().detach()), dim=1)
            self.logger.experiment.add_image('val/result', result_grid.type(torch.float32) / 255., self.global_step)
        return self.__package_result(mean_iou, mean_precision, mean_recall, mean_loss, mean_tp, mean_tn, mean_fp, mean_fn, mean_ca, commit=False)

    def validation_epoch_end(self, validation_step_outputs):
        logger = self.logger.experiment
        self.log('val/average_precision', torch.tensor([o["precision"] for o in validation_step_outputs], dtype=torch.float64).mean())
        self.log('val/average_recall', torch.tensor([o["recall"] for o in validation_step_outputs], dtype=torch.float64).mean())
        self.log('val/average_loss', torch.tensor([o["loss"] for o in validation_step_outputs], dtype=torch.float64).mean())
        self.log('val/average_ca', torch.tensor([o["ca"] for o in validation_step_outputs], dtype=torch.float64).mean())
        self.log('val/average_IoU', torch.tensor([o["iou"] for o in validation_step_outputs], dtype=torch.float64).mean(), prog_bar=True)
        self.__package_result(
            torch.tensor([o["iou"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            torch.tensor([o["precision"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            torch.tensor([o["recall"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            torch.tensor([o["loss"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            torch.tensor([o["tp"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            torch.tensor([o["tn"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            torch.tensor([o["fp"] for o in validation_step_outputs], dtype=torch.float64).mean(),
            torch.tensor([o["fn"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            torch.tensor([o["ca"] for o in validation_step_outputs], dtype=torch.float64).mean(), 
            commit=True)

    def test_step(self, batch_data, batch_idx):
        images = [x["image"] for x in batch_data]
        gt_instances = [x["instances"] for x in batch_data]
        logger = self.logger.experiment
        logits = self.network(batch_data)
        pred_instances = [x["instances"] for x in logits]
        images = torch.stack(images) 
        ious = []
        precision = []
        recall = []
        loss = []
        tp = [] 
        tn = []
        fp = []
        fn = []
        ca = []
        for gt_inst, pred_inst in zip(gt_instances, pred_instances):
            pred_count, gt_count = self.__count_instances(pred_instances, gt_instances)
            ca.append( 1 - torch.abs(pred_count - gt_count) / (gt_count + 1e-6))
            iou = self.__compute_iou_for_instances(gt_inst, pred_inst)
            ious.append(torch.tensor(iou).mean())
            p,r,l, TP, TN, FP, FN = self.__compute_precision_recall_loss_confusion(gt_inst, pred_inst)
            precision.append(p)
            recall.append(r)
            loss.append(l)
            tp.append(TP)
            tn.append(TN)
            fp.append(FP)
            fn.append(FN)
        mean_iou = torch.tensor(ious).mean()
        mean_precision = torch.tensor(precision).mean()
        mean_recall = torch.tensor(recall).mean()
        mean_loss = torch.tensor(loss).mean()
        mean_tp = torch.tensor(tp).mean()
        mean_tn = torch.tensor(tn).mean()
        mean_fp = torch.tensor(fp).mean()
        mean_fn = torch.tensor(fn).mean()
        mean_ca = torch.tensor(ca).mean()
        
        images = images.squeeze().detach().cpu().numpy().transpose(1,2,0)
        rgb_vis = images[:,:,:3].astype(np.uint8)
        thr_vis = np.repeat(images[:,:,[-1]], 3, axis=2).astype(np.uint8)

        pred_vis = np.zeros(tuple(rgb_vis.shape[:-1]), dtype=np.uint8)
        for batch_idx, instance in enumerate(pred_instances):
            masks = instance.pred_masks.detach().cpu()
            for i, mask in enumerate(masks):
                pred_vis[mask > 0] = (i + 1) + 128  # Assign a unique label for each instance
        pred_vis  = np.stack((pred_vis,) * 3, axis=-1) 
        gt_vis = np.zeros(tuple(rgb_vis.shape[:-1]), dtype=np.uint8)
        for batch_idx, instance in enumerate(gt_instances):
            masks = instance.gt_masks.detach().cpu()
            for i, mask in enumerate(masks):
                gt_vis[mask > 0] = 255  # Don't a unique label for each instance
        gt_vis  = np.stack((gt_vis,) * 3, axis=-1) 
        
        cv2.imwrite(os.path.join(self.output_dir, "rgb", "{:05}.png".format(self.counter)), cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.output_dir, "thr", "{:05}.png".format(self.counter)), cv2.cvtColor(thr_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.output_dir, "pred", "{:05}.png".format(self.counter)), cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.output_dir, "gt", "{:05}.png".format(self.counter)), cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))
        self.counter += 1
        return self.__package_result(mean_iou, mean_precision, mean_recall, mean_loss, mean_tp, mean_tn, mean_fp, mean_fn, mean_ca, commit=False)

    def test_epoch_end(self, test_step_outputs):
        # Here we just reuse the validation_epoch_end for testing
        return self.validation_epoch_end(test_step_outputs)
