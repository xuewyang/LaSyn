# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math, pdb

import torch


class Search(object):

    def __init__(self, tgt_dict):
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None
        self.pos_buf = None
        self.pos_beam_buf = None

    def _init_buffers(self, t):
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)
            self.pos_buf = torch.LongTensor().to(device=t.device)
            self.pos_beam_buf = torch.LongTensor().to(device=t.device)

    def step(self, step, lprobs, scores, beam_size):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


# class BeamSearch(Search):
#
#     def __init__(self, tgt_dict):
#         super().__init__(tgt_dict)
#
#     def step(self, step, lprobs, scores):   # [96, 3, 6632],
#         super()._init_buffers(lprobs)
#         bsz, beam_size, pos_size, vocab_size = lprobs.size()
#
#         if step == 0:
#             # at the first step all hypotheses are equally likely, so use
#             # only the first beam
#             lprobs = lprobs[:, ::beam_size, :, :].contiguous()
#         else:
#             # make probs contain cumulative scores for each hypothesis
#             lprobs.add_(scores[:, :, step - 1].unsqueeze(-1).unsqueeze(-1))
#         # pdb.set_trace()
#         torch.topk(
#             lprobs.view(bsz, -1),   # ([48, 1, 40, 6632]->[48, 6632*40]) -> ([48, 5, 40, 6632]->[48, 33160*40])
#             k=min(
#                 # Take the best 2 x beam_size predictions. We'll choose the first
#                 # beam_size of these which don't predict eos to continue with.
#                 beam_size * 2,
#                 lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
#             ),
#             out=(self.scores_buf, self.indices_buf),
#         )
#         # pdb.set_trace()
#         scores2, indices2 = torch.topk(lprobs.view(lprobs.size(0), lprobs.size(1), -1), k=beam_size * 2)
#         scores3, indices3 = torch.topk(scores2.view(bsz, -1), k=beam_size * 2)
#         ps = torch.div(indices2, vocab_size)
#         indices4 = torch.fmod(indices2, vocab_size)
#         ps2 = self.indices_buf.new(self.indices_buf.size(0), self.indices_buf.size(1)).fill_(0)
#         beam = torch.div(indices3, beam_size * 2)
#         indi = indices3.fmod(beam_size * 2)
#         indices5 = self.indices_buf.new(self.indices_buf.size(0), self.indices_buf.size(1)).fill_(0)
#         for ii in range(self.indices_buf.size(0)):
#             for jj in range(self.indices_buf.size(1)):
#                 indices5[ii][jj] = indices4[ii][beam[ii][jj]][indi[ii][jj]]
#                 ps2[ii][jj] = ps[ii][beam[ii][jj]][indi[ii][jj]]
#
#         self.scores_buf = scores3
#         self.indices_buf = indices5
#         self.beams_buf = beam
#         self.pos_buf = ps2
#         return self.scores_buf, self.indices_buf, self.beams_buf, self.pos_buf    # [48, 10(5x2)]

#
class BeamSearch(Search):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def step(self, step, lprobs, scores):   # [96, 3, 6632],
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))
        torch.topk(
            lprobs.view(bsz, -1),   # ([48, 1, 40, 6632]->[48, 6632*40]) -> ([48, 5, 40, 6632]->[48, 33160*40])
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)      # to check the tops belongs to which beam (0,1,2,3,4?)
        return self.scores_buf, self.indices_buf, self.beams_buf    # [48, 10(5x2)]


class BeamSearchNEM(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def step(self, step, lprobs, scores):   # [96, 3, 6632],
        super()._init_buffers(lprobs)
        bsz, beam_size, pos_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1).unsqueeze(-1))
        # pdb.set_trace()
        torch.topk(
            lprobs.view(bsz, -1),   # ([48, 1, 40, 6632]->[48, 6632*40]) -> ([48, 5, 40, 6632]->[48, 33160*40])
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        # pdb.set_trace()
        scores2, indices2 = torch.topk(lprobs.view(lprobs.size(0), lprobs.size(1), -1), k=beam_size * 2)
        scores3, indices3 = torch.topk(scores2.view(bsz, -1), k=beam_size * 2)
        ps = torch.div(indices2, vocab_size)
        indices4 = torch.fmod(indices2, vocab_size)
        ps2 = self.indices_buf.new(self.indices_buf.size(0), self.indices_buf.size(1)).fill_(0)
        beam = torch.div(indices3, beam_size * 2)
        indi = indices3.fmod(beam_size * 2)
        indices5 = self.indices_buf.new(self.indices_buf.size(0), self.indices_buf.size(1)).fill_(0)
        for ii in range(self.indices_buf.size(0)):
            for jj in range(self.indices_buf.size(1)):
                indices5[ii][jj] = indices4[ii][beam[ii][jj]][indi[ii][jj]]
                ps2[ii][jj] = ps[ii][beam[ii][jj]][indi[ii][jj]]

        self.scores_buf = scores3
        self.indices_buf = indices5
        self.beams_buf = beam
        self.pos_buf = ps2
        return self.scores_buf, self.indices_buf, self.beams_buf, self.pos_buf    # [48, 10(5x2)]


class LengthConstrainedBeamSearch(Search):

    def __init__(self, tgt_dict, min_len_a, min_len_b, max_len_a, max_len_b):
        super().__init__(tgt_dict)
        self.min_len_a = min_len_a
        self.min_len_b = min_len_b
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.beam = BeamSearch(tgt_dict)

    def step(self, step, lprobs, scores):
        min_lens = self.min_len_a * self.src_lengths + self.min_len_b
        max_lens = self.max_len_a * self.src_lengths + self.max_len_b
        lprobs[step < min_lens, :, self.eos] = -math.inf
        lprobs[step == max_lens, :, self.eos] = 0
        lprobs[step > max_lens, :, self.eos] = -math.inf
        return self.beam.step(step, lprobs, scores)


class DiverseBeamSearch(Search):
    """Diverse Beam Search.

    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.

    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    """

    def __init__(self, tgt_dict, num_groups, diversity_strength):
        super().__init__(tgt_dict)
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.diversity_buf = None
        self.beam = BeamSearch(tgt_dict)

    def step(self, step, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError(
                'DiverseBeamSearch requires --beam to be divisible by the number of groups'
            )

        # initialize diversity penalty
        if self.diversity_buf is None:
            self.diversity_buf = lprobs.new()
        torch.zeros(lprobs[:, 0, :].size(), out=self.diversity_buf)

        scores_G, indices_G, beams_G = [], [], []
        for g in range(self.num_groups):
            lprobs_g = lprobs[:, g::self.num_groups, :]
            scores_g = scores[:, g::self.num_groups, :] if step > 0 else None

            # apply diversity penalty
            if g > 0:
                lprobs_g = torch.add(lprobs_g, self.diversity_strength, self.diversity_buf.unsqueeze(1))
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, indices_buf, beams_buf = self.beam.step(step, lprobs_g, scores_g)
            beams_buf.mul_(self.num_groups).add_(g)

            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())

            # update diversity penalty
            self.diversity_buf.scatter_add_(
                1,
                indices_buf,
                self.diversity_buf.new_ones(indices_buf.size())
            )

        # interleave results from different groups
        self.scores_buf = torch.stack(scores_G, dim=2, out=self.scores_buf).view(bsz, -1)
        self.indices_buf = torch.stack(indices_G, dim=2, out=self.indices_buf).view(bsz, -1)
        self.beams_buf = torch.stack(beams_G, dim=2, out=self.beams_buf).view(bsz, -1)
        return self.scores_buf, self.indices_buf, self.beams_buf


class Sampling(Search):

    def __init__(self, tgt_dict, sampling_topk=-1, sampling_temperature=1.):
        super().__init__(tgt_dict)
        self.sampling_topk = sampling_topk
        self.sampling_temperature = sampling_temperature

    def step(self, step, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()
        pdb.set_trace()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            pdb.set_trace()
            lprobs = lprobs[:, ::beam_size, :].contiguous()

        # we exclude the first two vocab items, one of which is pad
        assert self.pad <= 1, 'sampling assumes the first two symbols can be ignored'
        lprobs_nopad = lprobs[:, :, 2:]
        pdb.set_trace()

        # only sample from top-k candidates
        if self.sampling_topk > 0:
            pdb.set_trace()
            lprobs_nopad, topk_indices = lprobs_nopad.topk(self.sampling_topk)

        # sampling temperature
        if self.sampling_temperature != 1.:
            lprobs_nopad = lprobs_nopad.div_(self.sampling_temperature)
        pdb.set_trace()
        # sample
        probs_nopad = lprobs_nopad.exp_()
        pdb.set_trace()
        if step == 0:
            pdb.set_trace()
            self.indices_buf = torch.multinomial(
                probs_nopad.view(bsz, -1),
                beam_size,
                replacement=True,
                out=self.indices_buf,
            ).view(bsz, beam_size)
        else:
            pdb.set_trace()
            self.indices_buf = torch.multinomial(
                probs_nopad.view(bsz * beam_size, -1),
                1,
                replacement=True,
                out=self.indices_buf,
            ).view(bsz, beam_size)

        if step == 0:
            # expand to beam size
            pdb.set_trace()
            probs_nopad = probs_nopad.expand(bsz, beam_size, -1)
        pdb.set_trace()
        # gather scores
        torch.gather(
            probs_nopad,
            dim=2,
            index=self.indices_buf.unsqueeze(-1),
            out=self.scores_buf,
        )
        pdb.set_trace()
        self.scores_buf = self.scores_buf.log_().view(bsz, -1)

        # remap indices if using top-k sampling
        pdb.set_trace()
        if self.sampling_topk > 0:
            self.indices_buf = torch.gather(
                topk_indices.expand(bsz, beam_size, -1),
                dim=2,
                index=self.indices_buf.unsqueeze(-1),
            ).squeeze(2)

        # remap indices since we excluded the first two vocab items
        self.indices_buf.add_(2)

        if step == 0:
            pdb.set_trace()
            self.beams_buf = self.indices_buf.new_zeros(bsz, beam_size)
        else:
            pdb.set_trace()
            self.beams_buf = torch.arange(0, beam_size, out=self.beams_buf).repeat(bsz, 1)
            # make scores cumulative
            self.scores_buf.add_(
                torch.gather(
                    scores[:, :, step - 1],
                    dim=1,
                    index=self.beams_buf,
                )
            )

        return self.scores_buf, self.indices_buf, self.beams_buf
