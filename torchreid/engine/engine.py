from __future__ import division, print_function, absolute_import

import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchreid import metrics
from torchreid.data.datasets import get_dataset_nickname
from torchreid.losses import deep_supervision
from torchreid.models.NAPE import NAPE
from torchreid.models.promptable_transformer_backbone import PromptableTransformerBackbone
from torchreid.utils import (
    re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results, Logger, AverageMeter, perc, plot_pairs_distance_distribution
)
from torchreid.utils.torchtools import collate


class Engine(object):
    def __init__(self, config, datamanager, writer, engine_state, use_gpu=True, save_model_flag=False,
                 detailed_ranking=False):
        self.config = config
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.save_model_flag = save_model_flag
        self.detailed_ranking = detailed_ranking

        self.engine_state = engine_state
        self.writer = writer
        self.logger = Logger.current_logger()

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(self, name='model', model=None, optim=None, sched=None):
        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, cmc, mAP, ssmd, save_dir, is_best=False):
        if self.save_model_flag:
            names = self.get_model_names()
            for name in names:
                save_checkpoint(
                    {
                        'state_dict': self._models[name].state_dict(),
                        'epoch': epoch + 1,
                        'rank1': cmc,
                        'mAP': mAP,
                        'ssmd': ssmd,
                        'config': self.config,
                        'optimizer': self._optims[name].state_dict(),
                        'scheduler': self._scheds[name].state_dict()
                    },
                    osp.join(save_dir, self.writer.model_name + name),
                    job_id=self.config.project.job_id,
                    is_best=is_best
                )

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()
        self.engine_state.update_lr(self.get_current_lr())

    def run(
            self,
            save_dir='log',
            fixbase_epoch=0,
            open_layers=None,
            test_only=False,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            visrank_q_idx_list=[],
            visrank_count=10,
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False,
            save_features=False
    ):
        """A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
            save_features (bool, optional): save test query and test gallery extracted features to disk
        """

        if test_only:
            self.test(
                0,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                visrank_q_idx_list=visrank_q_idx_list,
                visrank_count=visrank_count,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                save_features=save_features
            )
            return

        self.writer.total_run_timer.start()
        self.engine_state.estimated_num_batches = len(self.train_loader)
        self.engine_state.update_lr(self.get_current_lr())
        print('=> Start training')
        self.engine_state.training_started()
        mAP = 0

        for epoch in range(self.engine_state.start_epoch, self.engine_state.max_epoch):
            self.writer.epoch_timer.start()
            self.engine_state.epoch_started()

            self.train(
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )

            self.writer.epoch_timer.stop()
            self.engine_state.epoch_completed()

            if self.writer.intermediate_evaluate():
                print('=> Intermediate test')
                rank_1, mAP, ssmd = self.test(
                    epoch,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=False,
                    visrank_topk=visrank_topk,
                    visrank_q_idx_list=visrank_q_idx_list,
                    visrank_count=visrank_count,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    evalate_on_sources_only=True
                )
                self.save_model(epoch, rank_1, mAP, ssmd, save_dir)

        self.engine_state.training_completed()

        if self.engine_state.max_epoch > 0:
            print('=> Final test')
            rank_1, mAP, ssmd = self.test(
                self.engine_state.epoch,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                visrank_q_idx_list=visrank_q_idx_list,
                visrank_count=visrank_count,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                save_features=save_features,
                evalate_on_sources_only=False
            )
            self.save_model(self.engine_state.epoch, rank_1, mAP, ssmd, save_dir)

        self.writer.total_run_timer.stop()
        self.engine_state.run_completed()
        self.logger.close()

        return mAP

    def train(self, fixbase_epoch=0, open_layers=None):
        self.set_model_mode('train')
        self.logger.add_scalar('Train/lr', self.get_current_lr(), self.engine_state.epoch)

        self.two_stepped_transfer_learning(
            self.engine_state.epoch, fixbase_epoch, open_layers
        )

        self.writer.data_loading_timer.start()
        for self.batch_idx, data in enumerate(self.train_loader):
            self.writer.data_loading_timer.stop()
            self.writer.batch_timer.start()

            # —— 正常一次前向/反传 ——
            loss, loss_summary = self.forward_backward(data)
            self.writer.batch_timer.stop()

            self.writer.losses.update(loss_summary)
            self.writer.loss.update(loss)

            # ===== JSON batch 额外训练步：只动 engine，不改 DataLoader =====
            try:
                rep = int(getattr(self.config.data, "json_repeat_factor", 1))
            except Exception:
                rep = 1

            if rep > 1:
                def _to_tensor(x):
                    import torch
                    import numpy as np
                    if isinstance(x, torch.Tensor):
                        return x
                    if isinstance(x, np.ndarray):
                        return torch.from_numpy(x)
                    if isinstance(x, (list, tuple)):
                        try:
                            return torch.tensor(x)
                        except Exception:
                            return None
                    return None

                def _batch_json_ratio(d):
                    import torch
                    # 1) 优先用 has_json（布尔）
                    t = _to_tensor(d.get('has_json', None))
                    if t is not None:
                        try:
                            return float(t.float().mean().item())
                        except Exception:
                            pass

                    # 2) 再看 kp_quality（阈值判断）
                    thr = 0.20
                    try:
                        thr = float(getattr(self.config.model.nape, "kp_quality_threshold", 0.20))
                    except Exception:
                        pass
                    t = _to_tensor(d.get('kp_quality', None))
                    if t is not None:
                        try:
                            return float((t >= thr).float().mean().item())
                        except Exception:
                            pass

                    # 3) 最后看 prompt_masks（是否有非零像素）
                    t = _to_tensor(d.get('prompt_masks', None))
                    if t is not None:
                        try:
                            if t.dim() >= 3 and t.size(0) > 0:
                                flat = t.view(t.size(0), -1).float()
                                return float((flat.sum(dim=1) > 0).float().mean().item())
                        except Exception:
                            pass
                    return 0.0

                ratio = _batch_json_ratio(data)  # 0~1
                extra = int(round((rep - 1) * ratio))
                if extra < 0: extra = 0
                if extra > (rep - 1): extra = rep - 1

                for _ in range(extra):
                    self.writer.batch_timer.start()
                    loss2, loss_summary2 = self.forward_backward(data)
                    self.writer.batch_timer.stop()
                    self.writer.losses.update(loss_summary2)
                    self.writer.loss.update(loss2)
            # ===== JSON 额外训练步结束 =====

            self.writer.data_loading_timer.start()
            self.engine_state.batch_completed()

        self.update_lr()

    def forward_backward(self, data):
        raise NotImplementedError

    def test(
            self,
            epoch,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            visrank_q_idx_list=[],
            visrank_count=10,
            save_dir='',
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False,
            save_features=False,
            evalate_on_sources_only=False
    ):
        """Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        # 注释掉数据限制逻辑，使用完整数据集进行测试
        # print("正在限制测试数据量...")
        # import torch.utils.data
        # import numpy as np

        # 使用完整数据集进行测试，注释掉数据限制逻辑
        print("使用完整数据集进行测试...")

        self.writer.test_timer.start()

        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())
        if len(targets) == 0:
            raise RuntimeError("Test set is either empty or target dataset was not specified.")
        cmc_avg = AverageMeter()
        mAP_avg = AverageMeter()
        ssmd_avg = AverageMeter()
        pxl_acc_avg = AverageMeter()
        # TODO: capture metrics with Pandas frame (more scalable for new metrics)

        cmc_per_dataset = {}
        mAP_per_dataset = {}
        ssmd_per_dataset = {}
        pxl_acc_per_dataset = {}
        for name in targets:
            is_source_dataset = name in self.datamanager.sources
            domain = 'source' if is_source_dataset else 'target'
            if is_source_dataset or not evalate_on_sources_only:
                print('##### Evaluating {} ({}) #####'.format(name, domain))
                query_loader = self.test_loader[name]['query']
                gallery_loader = self.test_loader[name]['gallery']
                cmc, mAP, ssmd, avg_pxl_pred_accuracy = self._evaluate(
                    epoch,
                    dataset_name=name,
                    query_loader=query_loader,
                    gallery_loader=gallery_loader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    visrank_q_idx_list=visrank_q_idx_list,
                    visrank_count=visrank_count,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    rerank=rerank,
                    save_features=save_features
                )
                dataset_nickname = get_dataset_nickname(name)
                self.writer.report_performance(cmc, mAP, ssmd, avg_pxl_pred_accuracy, dataset_nickname)

                cmc_per_dataset[dataset_nickname] = perc(cmc)
                mAP_per_dataset[dataset_nickname] = perc(mAP)
                ssmd_per_dataset[dataset_nickname] = np.around(ssmd, 2)
                pxl_acc_per_dataset[dataset_nickname] = avg_pxl_pred_accuracy

                if is_source_dataset:
                    cmc_avg.update(cmc)
                    mAP_avg.update(mAP)
                    ssmd_avg.update(ssmd)
                    pxl_acc_avg.update(avg_pxl_pred_accuracy)
            else:
                print('##### Skipping {} ({}) #####'.format(name, domain))

        average_score_key = 'avg'
        cmc_per_dataset[average_score_key] = np.array(list(cmc_per_dataset.values())).mean(0)
        # transform dataset->cmc to cmc->dataset
        cmc_per_dataset = [{k: v[i - 1] for k, v in cmc_per_dataset.items()} for i in ranks]
        mAP_per_dataset[average_score_key] = np.array(list(mAP_per_dataset.values())).mean()
        ssmd_per_dataset[average_score_key] = np.array(list(ssmd_per_dataset.values())).mean()
        pxl_acc_per_dataset[average_score_key] = np.array(list(pxl_acc_per_dataset.values())).mean()

        self.engine_state.test_completed()

        self.writer.test_timer.stop()

        if mAP_avg.count != 0:
            self.writer.report_performance(cmc_avg.avg, mAP_avg.avg, ssmd_avg.avg, pxl_acc_avg.avg)
        self.writer.report_global_performance(cmc_per_dataset,
                                              mAP_per_dataset,
                                              ssmd_per_dataset,
                                              pxl_acc_per_dataset)
        r1 = cmc_avg.avg[0] if mAP_avg.count != 0 else 0
        return r1, mAP_avg.avg, ssmd_avg.avg

    @torch.no_grad()
    def _evaluate(
            self,
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            dataset_name,
            q_anns=None,
            g_anns=None
    ):
        print('Computing CMC and mAP ...')

        # ---------- 安全获取 eval_metric ----------
        eval_metric = "euclidean"  # 默认值
        try:
            dataset_obj = self.datamanager.test_loader[dataset_name]['query'].dataset
            # 如果 dataset 是 list，直接跳过
            if isinstance(dataset_obj, list):
                eval_metric = "euclidean"
            else:
                while hasattr(dataset_obj, 'dataset'):
                    dataset_obj = dataset_obj.dataset
                eval_metric = getattr(dataset_obj, "eval_metric", "euclidean")
        except Exception as e:
            print(f"[WARN] Cannot get eval_metric from dataset, fallback to euclidean. ({e})")
            eval_metric = "euclidean"

        # ---------- 调用 metrics.evaluate_rank ----------
        eval_metrics = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_anns=q_anns,
            g_anns=g_anns,
            eval_metric=eval_metric
        )

        cmc, mAP, ssmd, avg_pxl_pred_accuracy = eval_metrics
        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in [1, 5, 10]:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        return cmc, mAP, ssmd, avg_pxl_pred_accuracy

    def _feature_extraction(self, data_loader):
        f_, pids_, camids_ = [], [], []
        anns = []
        for batch_idx, data in enumerate(data_loader):
            imgs, pids, camids = self.parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            self.writer.test_batch_timer.start()
            features = self.extract_features(imgs)
            self.writer.test_batch_timer.stop()
            features = features.data.cpu()
            f_.append(features)
            pids_.extend(pids)
            camids_.extend(camids)
            anns.append(data)
        anns = collate(anns)
        f_ = torch.cat(f_, 0)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return f_, pids_, camids_, anns

    def compute_loss(self, criterion, outputs, targets, **kwargs):
        if isinstance(outputs, (tuple, list)):
            loss = deep_supervision(criterion, outputs, targets, **kwargs)
        else:
            loss = criterion(outputs, targets, **kwargs)
        return loss

    def extract_features(self, input):
        return self.model(input)

    def parse_data_for_train(self, data):
        imgs = data['image']
        pids = data['pid']
        return imgs, pids

    def parse_data_for_eval(self, data):
        imgs = data['image']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids

    def two_stepped_transfer_learning(
            self, epoch, fixbase_epoch, open_layers, model=None
    ):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if isinstance(model, nn.DataParallel):
            model = model.module
        is_promptable_transformer = False
        if isinstance(model, KPR) and isinstance(model.backbone_appearance_feature_extractor,
                                                 PromptableTransformerBackbone):
            is_promptable_transformer = True

        if fixbase_epoch > 0:
            if (epoch + 1) <= fixbase_epoch and open_layers is not None:
                if is_promptable_transformer:  # TODO generic implementation, but "close_specified_layers" then "open_specified_layers"
                    print("* Only freeze masks_patch_embed (epoch: {}/{})".format(epoch + 1, fixbase_epoch))
                    model.backbone_appearance_feature_extractor.masks_patch_embed.eval()
                    for p in model.backbone_appearance_feature_extractor.masks_patch_embed.parameters():
                        p.requires_grad = False
                else:
                    print(
                        '* Only train {} (epoch: {}/{})'.format(
                            open_layers, epoch + 1, fixbase_epoch
                        )
                    )
                    open_specified_layers(model, open_layers)
            else:
                if is_promptable_transformer:
                    print("* Train all (epoch: {}/{})".format(epoch + 1, fixbase_epoch))
                    model.backbone_appearance_feature_extractor.masks_patch_embed.train()
                    for p in model.backbone_appearance_feature_extractor.masks_patch_embed.parameters():
                        p.requires_grad = True
                else:
                    open_all_layers(model)

    def normalize(self, features):
        return F.normalize(features, p=2, dim=-1)

