# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math, pdb, torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(src_tokens=sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'],
                           prev_output_tokens=sample['net_input']['prev_output_tokens'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # pdb.set_trace()
        lprobs = torch.log(net_output[0])
        lprobs = lprobs.view(-1, lprobs.size(-1))       # 5888, 6632
        target = model.get_targets(sample, net_output).view(-1, 1)  # (46, 128) -> 5888
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        # pdb.set_trace()
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }


@register_criterion('label_smoothed_cross_entropy_tri')
class LabelSmoothedCrossEntropyCriterionTri(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.lambda1 = args.lambda1

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--lambda1', default=1, type=float, metavar='L',
                            help='hyper P lambda')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # pdb.set_trace()
        net_output = model(**sample['net_input'])
        loss1, nll_loss, loss_c, nll_loss_c = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        loss = loss1 + self.lambda1 * loss_c
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss1': utils.item(loss1.data) if reduce else loss1.data,
            'loss_c': utils.item(loss_c.data) if reduce else loss_c.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'nll_loss_c': utils.item(nll_loss_c.data) if reduce else nll_loss_c.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output[0]).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        # pdb.set_trace()
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        lprobs_c = model.get_normalized_probs_c(net_output[1], log_probs=True)
        lprobs_c = lprobs_c.view(-1, lprobs_c.size(-1))
        target_c = model.get_targets_c(sample, net_output[1]).view(-1, 1)
        non_pad_mask_c = target_c.ne(self.padding_idx)
        nll_loss_c = -lprobs_c.gather(dim=-1, index=target_c)[non_pad_mask_c]
        smooth_loss_c = -lprobs_c.sum(dim=-1, keepdim=True)[non_pad_mask_c]
        if reduce:
            nll_loss_c = nll_loss_c.sum()
            smooth_loss_c = smooth_loss_c.sum()
        eps_i_c = self.eps / lprobs_c.size(-1)
        loss_c = (1. - self.eps) * nll_loss_c + eps_i_c * smooth_loss_c
        # pdb.set_trace()
        return loss, nll_loss, loss_c, nll_loss_c

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # pdb.set_trace()
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss1': sum(log.get('loss1', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss_c': sum(log.get('loss_c', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'nll_loss_c': sum(log.get('nll_loss_c', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }


@register_criterion('label_smoothed_cross_entropy_pos')
class LabelSmoothedCrossEntropyCriterionPOS(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.lambda1 = args.lambda1

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--lambda1', default=1, type=float, metavar='L',
                            help='hyper P lambda')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # pdb.set_trace()
        net_output = model(src_tokens=sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'],
                           prev_output_tokens=sample['net_input']['prev_output_tokens'],
                           prev_output_tokens_c=sample['net_input']['prev_output_tokens_c'],
                           tgt_tokens_c=sample['target_c'])

        loss1, nll_loss, loss_c, nll_loss_c = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        loss = loss1 + self.lambda1 * loss_c
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss1': utils.item(loss1.data) if reduce else loss1.data,
            'loss_c': utils.item(loss_c.data) if reduce else loss_c.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'nll_loss_c': utils.item(nll_loss_c.data) if reduce else nll_loss_c.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # get targets
        target_z = net_output[1][1].view(-1, net_output[1][1].size(2))     # (BXS, 1) or (BXS, Sampling_size(40))
        target = model.get_targets(sample, net_output[0]).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        # calculate probabilities and gamma
        Pz_x = net_output[1][0].view(-1, net_output[1][0].size(2)).gather(dim=-1, index=target_z)
        if net_output[0][0].size(2) != 1:
            Py_zx = net_output[0][0].view(-1, net_output[0][0].size(2), net_output[0][0].size(3)).\
                gather(dim=2, index=target.unsqueeze(1).repeat(1, net_output[0][0].size(2), 1)).squeeze(2)
            non_pad_mask = non_pad_mask.repeat(1, net_output[0][0].size(2))
        else:
            Py_zx = net_output[0][0].view(-1, net_output[0][0].size(-1)).gather(dim=-1, index=target)
        # gamma = self.e_step(Pz_x, Py_zx)

        # calculate loss
        # lprobs = gamma * (torch.log(Pz_x) + torch.log(Py_zx))
        lprobs = torch.log(Py_zx)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        nll_loss = -lprobs[non_pad_mask]
        # smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            # smooth_loss = smooth_loss.sum()
        # eps_i = self.eps / lprobs.size(-1)
        # loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        loss = nll_loss

        # calculate CE loss for pos-tagging part
        lprobs_c = torch.log(net_output[1][0])
        # pdb.set_trace()
        lprobs_c = lprobs_c.view(-1, lprobs_c.size(-1))
        target_c = model.get_targets_c(sample, net_output[1]).view(-1, 1)
        non_pad_mask_c = target_c.ne(self.padding_idx)
        nll_loss_c = -lprobs_c.gather(dim=-1, index=target_c)[non_pad_mask_c]
        smooth_loss_c = -lprobs_c.sum(dim=-1, keepdim=True)[non_pad_mask_c]
        if reduce:
            nll_loss_c = nll_loss_c.sum()
            smooth_loss_c = smooth_loss_c.sum()
        eps_i_c = self.eps / lprobs_c.size(-1)
        loss_c = (1. - self.eps) * nll_loss_c + eps_i_c * smooth_loss_c
        # pdb.set_trace()
        return loss, loss, loss_c, nll_loss_c

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # pdb.set_trace()
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss1': sum(log.get('loss1', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss_c': sum(log.get('loss_c', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'nll_loss_c': sum(log.get('nll_loss_c', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

    def e_step(self, Pz_x, Py_zx):
        gamma = Pz_x * Py_zx  # responsibility      # B, S, 40
        gamma = torch.div(gamma, torch.sum(gamma, dim=-1, keepdim=True))
        return gamma.detach()


@register_criterion('label_smoothed_cross_entropy_nem')
class LabelSmoothedCrossEntropyCriterionNEM(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.lambda1 = args.lambda1

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--lambda1', default=0.5, type=float, metavar='L',
                            help='hyper P lambda')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(src_tokens=sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'],
                           prev_output_tokens=sample['net_input']['prev_output_tokens'])

        loss1, nll_loss, loss_c, nll_loss_c = self.compute_loss(net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        loss = loss1 + self.lambda1 * loss_c
        # loss = loss1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss1': utils.item(loss1.data) if reduce else loss1.data,
            'loss_c': utils.item(loss_c.data) if reduce else loss_c.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'nll_loss_c': utils.item(nll_loss_c.data) if reduce else nll_loss_c.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample, reduce=True):
        # get targets
        target = sample['target'].view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        # calculate probabilities and gamma
        Pz_x = net_output[1].view(-1, net_output[1].size(2))
        Py_zx = net_output[0].view(-1, net_output[0].size(2), net_output[0].size(3)).\
            gather(dim=2, index=target.unsqueeze(1).repeat(1, net_output[0].size(2), 1)).squeeze(2)
        non_pad_mask = non_pad_mask.repeat(1, net_output[0].size(2))
        gamma = self.e_step(Pz_x, Py_zx)

        # calculate loss
        lprobs = gamma * (torch.log(Pz_x) + torch.log(Py_zx))
        nll_loss = -lprobs[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()

        # calculate CE loss for pos-tagging part
        lprobs_c = torch.log(net_output[1])
        lprobs_c = lprobs_c.view(-1, lprobs_c.size(-1))
        target_c = sample['target_c'].view(-1, 1)
        non_pad_mask_c = target_c.ne(self.padding_idx)
        nll_loss_c = -lprobs_c.gather(dim=-1, index=target_c)[non_pad_mask_c]
        # pdb.set_trace()
        if reduce:
            nll_loss_c = nll_loss_c.sum()
        return nll_loss, nll_loss, nll_loss_c, nll_loss_c

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # pdb.set_trace()
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss1': sum(log.get('loss1', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss_c': sum(log.get('loss_c', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'nll_loss_c': sum(log.get('nll_loss_c', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

    def e_step(self, Pz_x, Py_zx):
        gamma = Pz_x * Py_zx  # responsibility      # B, S, 40
        gamma = torch.div(gamma, torch.sum(gamma, dim=-1, keepdim=True))
        return gamma.detach()


@register_criterion('label_smoothed_cross_entropy_nem_noreg')
class LabelSmoothedCrossEntropyCriterionNEMNoREG(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(src_tokens=sample['net_input']['src_tokens'], src_lengths=sample['net_input']['src_lengths'],
                           prev_output_tokens=sample['net_input']['prev_output_tokens'])

        loss1, nll_loss, loss_c, nll_loss_c = self.compute_loss(net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        loss = loss1
        # loss = loss1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss1': utils.item(loss1.data) if reduce else loss1.data,
            'loss_c': utils.item(loss_c.data) if reduce else loss_c.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'nll_loss_c': utils.item(nll_loss_c.data) if reduce else nll_loss_c.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample, reduce=True):
        # get targets
        target = sample['target'].view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        # calculate probabilities and gamma
        Pz_x = net_output[1].view(-1, net_output[1].size(2))
        Py_zx = net_output[0].view(-1, net_output[0].size(2), net_output[0].size(3)).\
            gather(dim=2, index=target.unsqueeze(1).repeat(1, net_output[0].size(2), 1)).squeeze(2)
        non_pad_mask = non_pad_mask.repeat(1, net_output[0].size(2))
        gamma = self.e_step(Pz_x, Py_zx)

        # calculate loss
        lprobs = gamma * (torch.log(Pz_x) + torch.log(Py_zx))
        nll_loss = -lprobs[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()

        return nll_loss, nll_loss, nll_loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # pdb.set_trace()
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss1': sum(log.get('loss1', 0) for log in logging_outputs) / sample_size / math.log(2),
            'loss_c': sum(log.get('loss_c', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'nll_loss_c': sum(log.get('nll_loss_c', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

    def e_step(self, Pz_x, Py_zx):
        gamma = Pz_x * Py_zx  # responsibility      # B, S, 40
        gamma = torch.div(gamma, torch.sum(gamma, dim=-1, keepdim=True))
        return gamma.detach()