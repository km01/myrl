import torch


@torch.no_grad()
def compute_ret(vals, rews, dones, truncated, gamma=0.99, lamda=0.95, next_val=None):
    roll_len = vals.size(0)

    vals = vals.squeeze(-1)
    rets = torch.zeros_like(vals)

    if next_val is None:
        next_v = torch.zeros_like(vals[0])
        truncated = truncated.clone().detach()
        truncated[-1, :] = True

    else:
        next_v = next_val.squeeze(-1)

    if lamda is not None:   # gae
        gae = torch.zeros_like(next_v)
        for t in reversed(range(roll_len)):
            val, rew, done, trunc = vals[t], rews[t], dones[t], truncated[t]
            next_v.masked_fill_(done, 0.)
            gae.masked_fill_(done, 0.)
            delta = rew - val + gamma * next_v
            gae = delta + gamma * lamda * gae
            gae.masked_fill_(trunc, 0.)
            rets[t].copy_(gae + val)
            next_v.copy_(val)

    else:   # naive mc
        ret = next_v
        for t in reversed(range(roll_len)):
            val, rew, done, trunc = vals[t], rews[t], dones[t], truncated[t]
            ret.masked_fill_(done, 0.)
            ret = rew + gamma * ret
            ret.masked_fill_(trunc, 0.)
            ret += val.masked_fill(torch.logical_not(trunc), 0.)
            rets.copy_(ret)

    rets = rets.unsqueeze(-1)
    return rets



# @torch.no_grad()
# def compute_ret(vals, rews, dones, truncated, gamma=0.99, lamda=0.95, next_val=None):
#
#     vals = vals.squeeze(-1)
#
#     if next_val is None:
#         next_v = torch.zeros_like(vals[0])
#         truncated = truncated.clone().detach()
#         truncated[-1, :] = True
#
#     else:
#         next_v = next_val.squeeze(-1)
#
#     rets = []
#
#     if lamda is not None:
#         gae = torch.zeros_like(next_v)
#         for val, rew, done, trunc in zip(reversed(vals), reversed(rews), reversed(dones), reversed(truncated)):
#             next_v.masked_fill_(done, 0.)
#             gae.masked_fill_(done, 0.)
#
#             delta = rew - val + gamma * next_v
#             gae = delta + gamma * lamda * gae
#             gae.masked_fill_(trunc, 0.)
#
#             ret = gae + val
#             rets.append(ret)
#             next_v = val.clone()
#
#     else:
#         ret = next_v
#         for val, rew, done, trunc in zip(reversed(vals), reversed(rews), reversed(dones), reversed(truncated)):
#             ret.masked_fill_(done, 0.)
#             ret = rew + gamma * ret
#             ret.masked_fill_(trunc, 0.)
#             ret += val.masked_fill(torch.logical_not(trunc), 0.)
#             rets.append(ret)
#
#     rets.reverse()
#     rets = torch.stack(rets, dim=0)
#     rets = rets.unsqueeze(-1)
#     return rets
