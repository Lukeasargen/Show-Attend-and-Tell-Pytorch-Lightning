import torch

vocab_size = 8

beamk = 4
topk = 3

seq_scores = torch.rand(beamk, vocab_size)

print(f"{seq_scores=}")

# Take the topk samples from each beam
_, candidate_idxs = torch.topk(seq_scores, topk, dim=1)

print(f"{candidate_idxs=}")

# Add the adjustment and FLATTEN the candidates
adj_idx = torch.tensor([i*vocab_size for i in range(beamk)]).unsqueeze(1)
print(f"{adj_idx=}")
candidate_idxs = (candidate_idxs+adj_idx).reshape(-1)
print(f"{candidate_idxs=}")

# Uniformly sample the candidates without replacement
# choice_idx = torch.randperm(candidate_idxs.numel(), device=candidate_idxs.device)[:beamk]
choice_idx = torch.multinomial(torch.ones(candidate_idxs.numel()), beamk)
print(f"{choice_idx=}")
pred_idx = candidate_idxs[choice_idx]
print(f"{pred_idx=}")


